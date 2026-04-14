import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from "@google/genai";
import { 
  Send, 
  Upload, 
  FileText, 
  Trash2, 
  Loader2, 
  MessageSquare, 
  Info, 
  ThumbsUp, 
  ThumbsDown,
  ChevronRight,
  Database,
  History
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { chunkText, generateEmbeddings, findRelevantChunks, type DocumentChunk } from './lib/rag';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: DocumentChunk[];
  timestamp: Date;
  feedback?: 'up' | 'down';
}

interface Document {
  name: string;
  chunks: DocumentChunk[];
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress('Extracting text...');
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/extract', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Failed to extract text';
        try {
          const errorData = await response.json();
          errorMessage = errorData.details || errorData.error || errorMessage;
        } catch (e) {
          const text = await response.text();
          console.error('Server returned non-JSON error:', text);
          if (text.includes('<!doctype html>')) {
            errorMessage = 'Server configuration error: API route returned HTML instead of JSON. Please check server logs.';
          }
        }
        throw new Error(errorMessage);
      }
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        console.error('Unexpected response format:', text);
        throw new Error('Server returned unexpected response format. Please check server logs.');
      }

      const { text, fileName } = await response.json();
      
      setUploadProgress('Chunking and Embedding...');
      const chunks = chunkText(text, fileName);
      
      // Generate embeddings for each chunk
      const chunksWithEmbeddings = [];
      for (let i = 0; i < chunks.length; i++) {
        setUploadProgress(`Embedding chunk ${i + 1}/${chunks.length}...`);
        const embedding = await generateEmbeddings(ai, chunks[i].text);
        chunksWithEmbeddings.push({ ...chunks[i], embedding });
      }

      setDocuments(prev => [...prev, { name: fileName, chunks: chunksWithEmbeddings }]);
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to process document');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeDocument = (name: string) => {
    setDocuments(prev => prev.filter(doc => doc.name !== name));
  };

  const handleSend = async () => {
    if (!input.trim() || isThinking) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsThinking(true);

    try {
      const allChunks = documents.flatMap(doc => doc.chunks);
      
      let context = '';
      let relevantChunks: DocumentChunk[] = [];

      if (allChunks.length > 0) {
        relevantChunks = await findRelevantChunks(ai, input, allChunks);
        context = relevantChunks.map(c => `[Source: ${c.source}, Page: ${c.page}]\n${c.text}`).join('\n\n');
      }

      const historyContext = messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n');

      const prompt = `
You are a helpful and accurate RAG chatbot. Use the provided context and conversation history to answer the user's query.

CONSTRAINTS:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say "I don't know. The provided documents do not contain information about this."
3. Be concise and professional.
4. Maintain a coherent conversation based on the history.

CONVERSATION HISTORY:
${historyContext}

RETRIEVED CONTEXT:
${context || 'No documents uploaded yet.'}

USER QUERY:
${input}

ASSISTANT RESPONSE:`;

      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
      });

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.text || "I'm sorry, I couldn't generate a response.",
        sources: relevantChunks,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: "Error: Failed to get a response from the AI.",
        timestamp: new Date(),
      }]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleFeedback = async (messageId: string, feedback: 'up' | 'down') => {
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback } : m));
    
    const message = messages.find(m => m.id === messageId);
    if (!message) return;

    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messageId,
          feedback,
          query: messages.find((_, i) => messages[i+1]?.id === messageId)?.content,
          response: message.content,
          context: message.sources?.map(s => s.text).join('\n')
        }),
      });
    } catch (error) {
      console.error('Feedback error:', error);
    }
  };

  return (
    <div className="flex h-screen bg-[#E4E3E0] text-[#141414] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 border-r border-[#141414] flex flex-col bg-[#E4E3E0] z-10">
        <div className="p-6 border-bottom border-[#141414]">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-5 h-5" />
            <h1 className="font-serif italic text-lg font-medium">Knowledge Base</h1>
          </div>
          <p className="text-xs opacity-60 uppercase tracking-widest mb-6">RAG Chatbot Pro</p>
          
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 py-3 border border-[#141414] hover:bg-[#141414] hover:text-[#E4E3E0] transition-all duration-200 disabled:opacity-50"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="text-sm font-medium uppercase tracking-wider">Upload Document</span>
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept=".pdf,.txt"
          />
          {isUploading && (
            <div className="mt-2 text-[10px] font-mono opacity-70 animate-pulse">
              {uploadProgress}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <div className="text-[10px] font-mono opacity-40 uppercase tracking-widest mb-2 px-2">Loaded Documents</div>
          {documents.length === 0 ? (
            <div className="px-2 py-8 text-center border border-dashed border-[#141414]/20 rounded">
              <Info className="w-4 h-4 mx-auto mb-2 opacity-20" />
              <p className="text-xs opacity-40 italic">No documents loaded</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.name} className="group flex items-center justify-between p-3 border border-[#141414] hover:bg-[#141414]/5 transition-colors">
                <div className="flex items-center gap-3 overflow-hidden">
                  <FileText className="w-4 h-4 flex-shrink-0" />
                  <span className="text-xs font-medium truncate">{doc.name}</span>
                </div>
                <button 
                  onClick={() => removeDocument(doc.name)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-600 transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="p-4 border-t border-[#141414] bg-[#D8D7D4]">
          <div className="flex items-center gap-2 mb-2 opacity-60">
            <History className="w-3.5 h-3.5" />
            <span className="text-[10px] uppercase tracking-wider font-semibold">Stats</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 border border-[#141414]/20 rounded bg-[#E4E3E0]/50">
              <div className="text-[10px] opacity-50">Chunks</div>
              <div className="text-sm font-mono font-bold">{documents.reduce((acc, d) => acc + d.chunks.length, 0)}</div>
            </div>
            <div className="p-2 border border-[#141414]/20 rounded bg-[#E4E3E0]/50">
              <div className="text-[10px] opacity-50">Memory</div>
              <div className="text-sm font-mono font-bold">{messages.length} msgs</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative bg-[#F0EFED]">
        {/* Header */}
        <header className="h-16 border-b border-[#141414] flex items-center px-8 justify-between bg-[#E4E3E0]">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <h2 className="font-serif italic text-lg">Chat Session</h2>
          </div>
          <div className="text-[10px] font-mono opacity-50 uppercase tracking-widest">
            {new Date().toLocaleDateString()}
          </div>
        </header>

        {/* Messages */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-8 space-y-8 scroll-smooth"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center opacity-30 max-w-md mx-auto text-center">
              <MessageSquare className="w-12 h-12 mb-4" />
              <h3 className="font-serif italic text-xl mb-2">Ready to assist</h3>
              <p className="text-sm">Upload documents to the knowledge base and start asking questions. I will answer based strictly on the provided context.</p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div 
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex flex-col max-w-3xl",
                  msg.role === 'user' ? "ml-auto items-end" : "mr-auto items-start"
                )}
              >
                <div className={cn(
                  "px-6 py-4 border border-[#141414] relative",
                  msg.role === 'user' ? "bg-[#141414] text-[#E4E3E0]" : "bg-white text-[#141414]"
                )}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  
                  {/* Feedback for Assistant */}
                  {msg.role === 'assistant' && (
                    <div className="absolute -bottom-8 right-0 flex gap-2">
                      <button 
                        onClick={() => handleFeedback(msg.id, 'up')}
                        className={cn(
                          "p-1.5 border border-[#141414] transition-colors",
                          msg.feedback === 'up' ? "bg-[#141414] text-white" : "hover:bg-[#141414]/5"
                        )}
                      >
                        <ThumbsUp className="w-3 h-3" />
                      </button>
                      <button 
                        onClick={() => handleFeedback(msg.id, 'down')}
                        className={cn(
                          "p-1.5 border border-[#141414] transition-colors",
                          msg.feedback === 'down' ? "bg-red-600 text-white border-red-600" : "hover:bg-[#141414]/5"
                        )}
                      >
                        <ThumbsDown className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 w-full">
                    <div className="flex items-center gap-2 mb-2 opacity-40">
                      <ChevronRight className="w-3 h-3" />
                      <span className="text-[10px] uppercase tracking-widest font-bold">Sources</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {msg.sources.map((source, idx) => (
                        <div key={idx} className="p-3 bg-[#141414]/5 border border-[#141414]/10 rounded text-[10px] font-mono">
                          <div className="flex justify-between mb-1">
                            <span className="font-bold truncate max-w-[150px]">{source.source}</span>
                            <span className="opacity-50">PG {source.page}</span>
                          </div>
                          <p className="opacity-70 line-clamp-2 italic">"{source.text}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-2 text-[9px] font-mono opacity-30 uppercase tracking-tighter">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isThinking && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-start max-w-3xl"
            >
              <div className="px-6 py-4 border border-[#141414] bg-white/50 animate-pulse flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-xs font-mono italic">Synthesizing response...</span>
              </div>
            </motion.div>
          )}
        </div>

        {/* Input */}
        <div className="p-8 bg-[#E4E3E0] border-t border-[#141414]">
          <div className="max-w-4xl mx-auto relative">
            <input 
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder={documents.length > 0 ? "Ask a question about your documents..." : "Upload a document to start..."}
              disabled={isThinking}
              className="w-full pl-6 pr-16 py-4 bg-white border border-[#141414] focus:outline-none focus:ring-2 focus:ring-[#141414]/10 transition-all placeholder:italic placeholder:opacity-40"
            />
            <button 
              onClick={handleSend}
              disabled={!input.trim() || isThinking}
              className="absolute right-2 top-2 bottom-2 px-4 bg-[#141414] text-[#E4E3E0] hover:bg-[#141414]/90 disabled:opacity-30 transition-all flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-3 text-center">
            <p className="text-[9px] opacity-30 uppercase tracking-[0.2em]">Powered by Gemini 3 Flash & Vector Search</p>
          </div>
        </div>
      </main>
    </div>
  );
}
