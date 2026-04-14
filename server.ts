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

async function startServer() {
  const app = express();
  const PORT = parseInt(process.env.PORT || '3000', 10);

  app.use(express.json());

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // API Route for PDF extraction
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

  // Simple feedback storage (in-memory for this demo)
  const feedbackStore: any[] = [];
  app.post('/api/feedback', (req, res) => {
    const { messageId, feedback, query, response, context } = req.body;
    feedbackStore.push({
      id: Date.now(),
      messageId,
      feedback,
      query,
      response,
      context,
      timestamp: new Date().toISOString()
    });
    console.log('Feedback received:', feedback);
    res.json({ success: true });
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
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('Environment:', process.env.NODE_ENV || 'development');
    console.log('API Routes registered: /api/health, /api/extract, /api/feedback');
  });
}

startServer();
