import { GoogleGenAI } from "@google/genai";

export interface DocumentChunk {
  id: string;
  text: string;
  source: string;
  page?: number;
  embedding?: number[];
}

export function chunkText(text: string, source: string, chunkSize: number = 600, overlap: number = 100): DocumentChunk[] {
  const chunks: DocumentChunk[] = [];
  const words = text.split(/\s+/);
  
  let i = 0;
  while (i < words.length) {
    const chunkWords = words.slice(i, i + chunkSize);
    const chunkText = chunkWords.join(' ');
    
    chunks.push({
      id: crypto.randomUUID(),
      text: chunkText,
      source: source,
      // Simple page estimation if text has form feeds or similar, 
      // but usually pdf-parse doesn't give clean page breaks easily without more config
      page: Math.floor(i / 500) + 1 
    });
    
    i += (chunkSize - overlap);
  }
  
  return chunks;
}

export async function generateEmbeddings(ai: GoogleGenAI, text: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: "gemini-embedding-2-preview",
    contents: [{ parts: [{ text }] }],
  });
  return response.embeddings[0].values;
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function findRelevantChunks(ai: GoogleGenAI, query: string, chunks: DocumentChunk[], topK: number = 5): Promise<DocumentChunk[]> {
  if (chunks.length === 0) return [];
  
  const queryEmbedding = await generateEmbeddings(ai, query);
  
  const scoredChunks = chunks.map(chunk => ({
    chunk,
    score: chunk.embedding ? cosineSimilarity(queryEmbedding, chunk.embedding) : 0
  }));
  
  return scoredChunks
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(item => item.chunk);
}
