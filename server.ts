import 'dotenv/config';
import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { PDFParse } from 'pdf-parse';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const upload = multer({ storage: multer.memoryStorage() });

// Service URLs (local)
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const EMBED_SERVER_URL = process.env.EMBED_SERVER_URL || 'http://localhost:8000';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.2:3b';

async function startServer() {
  const app = express();
  const PORT = parseInt(process.env.PORT || '3000', 10);

  app.use(express.json({ limit: '50mb' }));

  // Health check
  app.get('/api/health', async (req, res) => {
    try {
      // Check Ollama
      const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/tags`).catch(() => null);
      const ollamaOk = ollamaRes?.ok ?? false;

      // Check Embedding server
      const embedRes = await fetch(`${EMBED_SERVER_URL}/health`).catch(() => null);
      const embedOk = embedRes?.ok ?? false;

      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        services: {
          ollama: ollamaOk ? 'connected' : 'disconnected',
          embedServer: embedOk ? 'connected' : 'disconnected',
        },
        ollamaModel: OLLAMA_MODEL,
      });
    } catch {
      res.json({ status: 'ok', timestamp: new Date().toISOString() });
    }
  });

  // ---------------------------------------------------------------
  // PDF Text Extraction (unchanged)
  // ---------------------------------------------------------------
  app.post('/api/extract', upload.single('file'), async (req, res) => {
    console.log('[/api/extract] Request received');
    try {
      if (!req.file) {
        console.error('[/api/extract] No file in request');
        return res.status(400).json({ error: 'No file uploaded' });
      }

      console.log(`[/api/extract] Processing file: ${req.file.originalname} (${req.file.mimetype})`);
      let text = '';
      const fileName = req.file.originalname;

      if (req.file.mimetype === 'application/pdf') {
        try {
          console.log('[/api/extract] Initializing PDFParse');
          const parser = new PDFParse({ data: req.file.buffer });
          console.log('[/api/extract] Extracting text...');
          const result = await parser.getText();
          text = result.text;
          console.log(`[/api/extract] Successfully extracted ${text.length} characters`);
          await parser.destroy();
        } catch (pdfError) {
          console.error('[/api/extract] PDF parsing failed:', pdfError);
          throw pdfError;
        }
      } else {
        text = req.file.buffer.toString('utf-8');
      }

      res.json({ text, fileName });
    } catch (error) {
      console.error('[/api/extract] Extraction error:', error);
      res.status(500).json({ 
        error: 'Failed to extract text from file',
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // ---------------------------------------------------------------
  // Chat via Ollama
  // ---------------------------------------------------------------
  app.post('/api/chat', async (req, res) => {
    console.log('[/api/chat] Request received');
    try {
      const { prompt, context } = req.body;

      if (!prompt) {
        return res.status(400).json({ error: 'prompt is required' });
      }

      const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: OLLAMA_MODEL,
          prompt: prompt,
          stream: false,
        }),
      });

      if (!ollamaRes.ok) {
        const errText = await ollamaRes.text();
        console.error('[/api/chat] Ollama error:', errText);
        throw new Error(`Ollama returned ${ollamaRes.status}: ${errText}`);
      }

      const data = await ollamaRes.json() as any;
      res.json({ response: data.response });
    } catch (error) {
      console.error('[/api/chat] Chat error:', error);
      res.status(500).json({
        error: 'Failed to get response from Ollama',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // ---------------------------------------------------------------
  // Embedding Proxies → Python FastAPI Server
  // ---------------------------------------------------------------

  // Text embedding
  app.post('/api/embed/text', async (req, res) => {
    try {
      const { text } = req.body;
      if (!text) {
        return res.status(400).json({ error: 'text is required' });
      }

      const embedRes = await fetch(`${EMBED_SERVER_URL}/embed/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!embedRes.ok) {
        const errText = await embedRes.text();
        throw new Error(`Embedding server error: ${errText}`);
      }

      const data = await embedRes.json();
      res.json(data);
    } catch (error) {
      console.error('[/api/embed/text] Error:', error);
      res.status(500).json({
        error: 'Failed to generate text embedding',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Image embedding
  app.post('/api/embed/image', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Forward the file to the Python server as multipart
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append('file', blob, req.file.originalname);

      const embedRes = await fetch(`${EMBED_SERVER_URL}/embed/image`, {
        method: 'POST',
        body: formData,
      });

      if (!embedRes.ok) {
        const errText = await embedRes.text();
        throw new Error(`Embedding server error: ${errText}`);
      }

      const data = await embedRes.json();
      res.json(data);
    } catch (error) {
      console.error('[/api/embed/image] Error:', error);
      res.status(500).json({
        error: 'Failed to generate image embedding',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Audio embedding
  app.post('/api/embed/audio', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Forward the file to the Python server as multipart
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append('file', blob, req.file.originalname);

      const embedRes = await fetch(`${EMBED_SERVER_URL}/embed/audio`, {
        method: 'POST',
        body: formData,
      });

      if (!embedRes.ok) {
        const errText = await embedRes.text();
        throw new Error(`Embedding server error: ${errText}`);
      }

      const data = await embedRes.json();
      res.json(data);
    } catch (error) {
      console.error('[/api/embed/audio] Error:', error);
      res.status(500).json({
        error: 'Failed to generate audio embedding',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Error handler to prevent HTML responses for API errors
  app.use('/api', (err: any, req: any, res: any, next: any) => {
    console.error('API Error:', err);
    res.status(500).json({ 
      error: 'Internal Server Error',
      details: err.message || String(err)
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('Environment:', process.env.NODE_ENV || 'development');
    console.log(`Ollama: ${OLLAMA_BASE_URL} (model: ${OLLAMA_MODEL})`);
    console.log(`Embedding Server: ${EMBED_SERVER_URL}`);
    console.log('API Routes: /api/health, /api/extract, /api/chat, /api/embed/*');
  });
}

startServer();
