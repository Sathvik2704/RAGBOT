/**
 * RAG (Retrieval-Augmented Generation) Library
 * 
 * Handles text chunking, embedding generation via local servers,
 * and cosine similarity retrieval across text/image/audio modalities.
 * All embeddings are 384-dimensional vectors in a shared semantic space.
 */

export interface DocumentChunk {
  id: string;
  text: string;
  source: string;
  page?: number;
  embedding?: number[];
  modality?: 'text' | 'image' | 'audio';
}

/**
 * Split text into overlapping chunks of ~chunkSize words.
 */
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
      page: Math.floor(i / 500) + 1,
      modality: 'text',
    });
    
    i += (chunkSize - overlap);
  }
  
  return chunks;
}

/**
 * Generate a 384-dim text embedding via the local embedding server.
 */
export async function generateEmbeddings(text: string): Promise<number[]> {
  const response = await fetch('/api/embed/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(`Text embedding failed: ${err.error || err.details || response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Generate a 384-dim image embedding via the local embedding server.
 */
export async function generateImageEmbedding(file: File): Promise<number[]> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/embed/image', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(`Image embedding failed: ${err.error || err.details || response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Generate a 384-dim audio embedding via the local embedding server.
 */
export async function generateAudioEmbedding(file: File): Promise<number[]> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/embed/audio', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(`Audio embedding failed: ${err.error || err.details || response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Cosine similarity between two vectors.
 */
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

/**
 * Find the most relevant chunks for a text query using cosine similarity.
 * Works across all modalities since they share the same 384-dim embedding space.
 */
export async function findRelevantChunks(query: string, chunks: DocumentChunk[], topK: number = 5): Promise<DocumentChunk[]> {
  if (chunks.length === 0) return [];
  
  const queryEmbedding = await generateEmbeddings(query);
  
  const scoredChunks = chunks.map(chunk => ({
    chunk,
    score: chunk.embedding ? cosineSimilarity(queryEmbedding, chunk.embedding) : 0
  }));
  
  return scoredChunks
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(item => item.chunk);
}
