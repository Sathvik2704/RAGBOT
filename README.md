# RAG Chatbot Pro — Local Multimodal (Ollama + Embeddings)

An advanced Retrieval-Augmented Generation chatbot with **multimodal** support (text, images, audio), source citations, and conversation memory — running **fully locally** with Ollama and a Python embedding server.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Browser (React Frontend)                    │
│  Upload: PDF/TXT/Image/Audio    Chat: text queries           │
└─────────┬──────────────────────────────────┬────────────────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Express Server (server.ts, port 3000)          │
│  /api/extract ──► PDF parsing (local)                       │
│  /api/embed/* ──► Proxy to Python server (port 8000)        │
│  /api/chat    ──► Proxy to Ollama (port 11434)              │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         ▼                                   ▼
┌────────────────────────┐  ┌───────────────────────────────┐
│  Python Embed Server   │  │         Ollama                │
│  (FastAPI, port 8000)  │  │    (port 11434)               │
│  multi-modal-embed-    │  │  llama3.2:3b (LLM)           │
│  small (120M params)   │  │  for chat/reasoning           │
└────────────────────────┘  └───────────────────────────────┘
```

## Prerequisites

- **Node.js** (v18+)
- **Python** (3.10+)
- **[Ollama](https://ollama.ai/)** installed and running

## Setup

### 1. Install Node.js dependencies
```bash
npm install
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the Ollama model
```bash
ollama pull llama3.2:3b
```

### 4. Configure environment
```bash
cp .env.example .env.local
# Edit .env.local with your Supabase credentials
```

### 5. Run the SQL migration
Run `schema.sql` in your Supabase SQL Editor.

## Running

Start all three services:

```bash
# Terminal 1: Python embedding server
python embed_server.py

# Terminal 2: Ollama (if not already running)
ollama serve

# Terminal 3: App
npm run dev
```

Open http://localhost:3000

## Supported File Types

| Type | Formats | Embedding |
|------|---------|-----------|
| Text | PDF, TXT | MiniLM-L6-v2 (384-dim) |
| Image | JPG, PNG, WebP | SigLIP-base (384-dim) |
| Audio | WAV, MP3, OGG, FLAC | Whisper-tiny (384-dim) |

All modalities share the same 384-dimensional embedding space, enabling **cross-modal search** (e.g., query "cat" to find a cat image).
