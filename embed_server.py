"""
Multimodal Embedding Server
FastAPI server that loads multi-modal-embed-small and exposes REST endpoints
for text, image, and audio embedding generation.

Model: llm-semantic-router/multi-modal-embed-small (~120M params)
- Text encoder: MiniLM-L6-v2 (22M params) → 384-dim vectors
- Image encoder: SigLIP-base-patch16-512 (86M params) → 384-dim vectors
- Audio encoder: Whisper-tiny encoder (8M params) → 384-dim vectors

All modalities share the same 384-dimensional embedding space.
"""

import io
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from scipy import signal
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
    WhisperModel,
    WhisperFeatureExtractor,
)
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definition (from the official model card)
# ---------------------------------------------------------------------------
class MultiModalEmbedder(nn.Module):
    """Standalone multimodal embedder matching the HF model card exactly."""

    def __init__(self):
        super().__init__()

        # Text encoder (384d, no projection needed)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Image encoder (768d -> 384d projection)
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-512"
        ).vision_model
        self.image_proj = nn.Linear(768, 384)

        # Audio encoder (384d, no projection needed)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny"
        )
        self.audio_encoder = WhisperModel.from_pretrained(
            "openai/whisper-tiny"
        ).encoder

    @torch.no_grad()
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def encode_image(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)  # 768 -> 384
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def encode_audio(self, waveform):
        # waveform: numpy array or tensor at 16kHz
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        inputs = self.audio_processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.audio_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model() -> MultiModalEmbedder:
    """Download and load the multi-modal-embed-small model with trained weights."""
    logger.info("Initializing MultiModalEmbedder...")
    model = MultiModalEmbedder()

    logger.info("Downloading trained weights from HuggingFace Hub...")
    checkpoint_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small",
        filename="model.pt",
    )

    logger.info("Loading state dict...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load text encoder weights
    model.text_encoder.load_state_dict(
        {
            k.replace("text_encoder.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.encoder.")
        }
    )

    # Load image encoder and projection weights
    model.image_encoder.load_state_dict(
        {
            k.replace("image_encoder.vision_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.vision_encoder.")
        }
    )
    model.image_proj.load_state_dict(
        {
            k.replace("image_encoder.projection.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.projection.")
        }
    )

    # Load audio encoder weights
    model.audio_encoder.load_state_dict(
        {
            k.replace("audio_encoder.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("audio_encoder.encoder.")
        }
    )

    model.eval()
    logger.info("Model loaded successfully!")
    return model


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
_model: MultiModalEmbedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    global _model
    _model = load_model()
    yield
    # Cleanup
    _model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


app = FastAPI(
    title="Multimodal Embedding Server",
    description="Generates 384-dim embeddings for text, images, and audio using multi-modal-embed-small",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Express server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "multi-modal-embed-small",
        "embedding_dim": 384,
        "modalities": ["text", "image", "audio"],
    }


@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(req: TextRequest):
    """Generate a 384-dim embedding for a text string."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        embedding = _model.encode_text(req.text)
        vec = embedding.squeeze().tolist()
        return EmbeddingResponse(embedding=vec, dimensions=len(vec))
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate a 384-dim embedding for an uploaded image (JPG/PNG/WebP)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"Expected image file, got {file.content_type}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = _model.encode_image(image)
        vec = embedding.squeeze().tolist()
        return EmbeddingResponse(embedding=vec, dimensions=len(vec))
    except Exception as e:
        logger.error(f"Image embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/audio", response_model=EmbeddingResponse)
async def embed_audio(file: UploadFile = File(...)):
    """Generate a 384-dim embedding for an uploaded audio file (WAV/MP3/OGG).
    Audio is resampled to 16kHz as required by the Whisper encoder.
    """
    allowed_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
    ]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, detail=f"Expected audio file, got {file.content_type}"
        )

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        # Load audio using soundfile (works on Windows without native deps)
        audio_data, sample_rate = sf.read(buffer, dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)

        # Convert to numpy array (already is, but ensure float32)
        waveform = audio_data.astype(np.float32)

        embedding = _model.encode_audio(waveform)
        vec = embedding.squeeze().tolist()
        return EmbeddingResponse(embedding=vec, dimensions=len(vec))
    except Exception as e:
        logger.error(f"Audio embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
