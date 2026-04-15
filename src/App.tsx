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
  History,
  LogOut,
  LogIn
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { chunkText, generateEmbeddings, findRelevantChunks, type DocumentChunk } from './lib/rag';
import { supabase } from './lib/supabase';

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
  id: string;
  name: string;
  chunks: DocumentChunk[];
  storagePath?: string;
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
    loadUserData();
  }, []);

  const loadUserData = async () => {
    try {
      // Fetch documents
      const { data: docsData, error: docsError } = await supabase
        .from('documents')
        .select('*')
        .order('created_at', { ascending: true });
        
      if (docsError) throw docsError;
      if (docsData) {
        setDocuments(docsData.map(d => ({
          id: d.id,
          name: d.file_name,
          chunks: d.chunks_data || [],
          storagePath: d.storage_path
        })));
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

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

      // We still use our local /api/extract to parse the PDF text
      const response = await fetch('/api/extract', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to extract text from file.');
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

      setUploadProgress('Storing file securely...');
      const fileExt = file.name.split('.').pop();
      const storagePath = `uploads/${crypto.randomUUID()}.${fileExt}`;
      
      const { error: uploadError } = await supabase.storage
        .from('user-documents')
        .upload(storagePath, file);

      if (uploadError) throw uploadError;

      setUploadProgress('Saving to knowledge base...');
      
      const { data: insertedDoc, error: dbError } = await supabase
        .from('documents')
        .insert({
          file_name: fileName,
          storage_path: storagePath,
          chunks_data: chunksWithEmbeddings
        })
        .select()
        .single();

      if (dbError) throw dbError;

      if (insertedDoc) {
        setDocuments(prev => [...prev, { 
          id: insertedDoc.id, 
          name: fileName, 
          chunks: chunksWithEmbeddings,
          storagePath
        }]);
      }
      
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to process document');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeDocument = async (id: string, storagePath?: string) => {
    // Optimistic UI update
    setDocuments(prev => prev.filter(doc => doc.id !== id));
    
    try {
      if (storagePath) {
        await supabase.storage.from('user-documents').remove([storagePath]);
      }
      await supabase.from('documents').delete().eq('id', id);
    } catch (error) {
      console.error("Failed to delete document:", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isThinking) return;

    const query = input;
    setInput('');
    setIsThinking(true);

    let userMsgId = crypto.randomUUID();

    try {
      const userMessage: Message = {
        id: userMsgId,
        role: 'user',
        content: query,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMessage]);

      const allChunks = documents.flatMap(doc => doc.chunks);
      let context = '';
      let relevantChunks: DocumentChunk[] = [];

      if (allChunks.length > 0) {
        relevantChunks = await findRelevantChunks(ai, query, allChunks);
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
${query}

ASSISTANT RESPONSE:`;

      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
      });

      const responseText = result.text || "I'm sorry, I couldn't generate a response.";

      let assistantMsgId = crypto.randomUUID();

      const assistantMessage: Message = {
        id: assistantMsgId,
        role: 'assistant',
        content: responseText,
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

  const handleFeedback = (messageId: string, feedback: 'up' | 'down') => {
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback } : m));
  };

  // -------------------------------------------------------------
  // RENDER LOGIC
  // -------------------------------------------------------------

  return (
    <div className="flex h-screen bg-[#E4E3E0] text-[#141414] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 border-r border-[#141414] flex flex-col bg-[#E4E3E0] z-10 flex-shrink-0">
        <div className="p-6 border-b border-[#141414]">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-5 h-5 text-blue-600" />
            <h1 className="font-serif text-xl tracking-tight">Knowledge Base</h1>
          </div>
          <p className="text-[10px] opacity-60 uppercase tracking-[0.2em] mb-6">LOCAL SESSION</p>
          
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 py-3 border border-[#141414] hover:bg-[#141414] hover:text-[#E4E3E0] transition-all duration-200 disabled:opacity-50"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="text-xs font-semibold uppercase tracking-wider">Upload Document</span>
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept=".pdf,.txt"
          />
          {isUploading && (
            <div className="mt-3 text-[10px] font-mono text-blue-600 animate-pulse text-center">
              {uploadProgress}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <div className="text-[10px] font-mono opacity-50 uppercase tracking-widest mb-4 px-2">Stored Documents</div>
          {documents.length === 0 ? (
            <div className="px-4 py-10 text-center border border-dashed border-[#141414]/20 rounded bg-white/30">
              <Info className="w-4 h-4 mx-auto mb-3 opacity-30" />
              <p className="text-xs opacity-50 italic">Your database is empty.</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.id} className="group flex items-center justify-between p-3 border border-[#141414] bg-white/50 hover:bg-[#141414] hover:text-[#E4E3E0] transition-colors cursor-default">
                <div className="flex items-center gap-3 overflow-hidden">
                  <FileText className="w-4 h-4 flex-shrink-0 opacity-70" />
                  <span className="text-xs font-medium truncate">{doc.name}</span>
                </div>
                <button 
                  onClick={() => removeDocument(doc.id, doc.storagePath)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-all focus:opacity-100"
                  title="Remove from DB"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="p-4 border-t border-[#141414] bg-[#D8D7D4]">
           <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 opacity-60">
                <History className="w-3.5 h-3.5" />
                <span className="text-[10px] uppercase tracking-wider font-semibold">Stats</span>
              </div>
           </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 border border-[#141414] bg-white/80">
              <div className="text-[10px] opacity-60">Memory Items</div>
              <div className="text-base font-serif italic">{messages.length}</div>
            </div>
            <div className="p-2 border border-[#141414] bg-white/80">
              <div className="text-[10px] opacity-60">Knowledge Size</div>
              <div className="text-base font-serif italic">{documents.length}</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative bg-[#F4F4F4]">
        {/* Header */}
        <header className="h-16 border-b border-[#141414] flex items-center px-8 justify-between bg-white/50 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
            <h2 className="font-serif italic text-xl">Operational Terminal</h2>
          </div>
          <div className="text-[10px] font-mono opacity-40 uppercase tracking-widest">
            {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'short', day: 'numeric' })}
          </div>
        </header>

        {/* Messages */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-8 space-y-8 scroll-smooth"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center opacity-40 max-w-md mx-auto text-center">
              <MessageSquare className="w-12 h-12 mb-6 text-blue-600 opacity-60" />
              <h3 className="font-serif italic text-2xl mb-3">System Online</h3>
              <p className="text-sm">Welcome back. Your conversation securely persists across sessions. Start by asking a question about your documents.</p>
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
                  "px-6 py-4 border border-[#141414] relative shadow-sm",
                  msg.role === 'user' ? "bg-[#141414] text-[#E4E3E0]" : "bg-white text-[#141414]"
                )}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  
                  {/* Feedback for Assistant */}
                  {msg.role === 'assistant' && (
                    <div className="absolute -bottom-8 right-0 flex gap-2">
                      <button 
                        onClick={() => handleFeedback(msg.id, 'up')}
                        title="Good Response"
                        className={cn(
                          "p-1.5 border border-[#141414] transition-colors rounded-none",
                          msg.feedback === 'up' ? "bg-[#141414] text-white" : "bg-white hover:bg-[#141414]/10"
                        )}
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      <button 
                        onClick={() => handleFeedback(msg.id, 'down')}
                        title="Poor Response"
                        className={cn(
                          "p-1.5 border border-[#141414] transition-colors rounded-none",
                          msg.feedback === 'down' ? "bg-red-600 text-white border-red-600" : "bg-white hover:bg-[#141414]/10"
                        )}
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 w-full pl-2">
                    <div className="flex items-center gap-2 mb-2 opacity-50">
                      <ChevronRight className="w-3 h-3" />
                      <span className="text-[10px] uppercase tracking-widest font-bold text-blue-800">Referenced Fragments</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {msg.sources.map((source, idx) => (
                        <div key={idx} className="p-3 bg-white border border-[#141414]/20 shadow-sm text-[10px] font-mono group hover:border-blue-600/50 transition-colors">
                          <div className="flex justify-between mb-2">
                            <span className="font-bold truncate max-w-[150px] text-blue-700">{source.source}</span>
                            <span className="opacity-50">PG {source.page}</span>
                          </div>
                          <p className="opacity-70 line-clamp-3 italic">"{source.text}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-2 text-[10px] font-mono opacity-40 uppercase tracking-widest text-[#141414]">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
              <div className="px-6 py-4 border border-[#141414] bg-white shadow-sm flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                <span className="text-xs font-mono italic opacity-70">Synthesizing response...</span>
              </div>
            </motion.div>
          )}
        </div>

        {/* Input */}
        <div className="p-8 bg-[#E4E3E0] border-t border-[#141414] z-10">
          <div className="max-w-4xl mx-auto relative group">
            <input 
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder={documents.length > 0 ? "Query the knowledge base..." : "Upload documents to begin..."}
              disabled={isThinking}
              className="w-full pl-6 pr-16 py-4 bg-white border border-[#141414] focus:outline-none focus:ring-2 focus:ring-blue-600/20 transition-all placeholder:italic placeholder:opacity-40 text-base shadow-sm"
            />
            <button 
              onClick={handleSend}
              disabled={!input.trim() || isThinking}
              className="absolute right-2 top-2 bottom-2 px-4 bg-[#141414] text-[#E4E3E0] hover:bg-blue-600 disabled:opacity-30 disabled:hover:bg-[#141414] transition-all flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-4 text-center">
            <p className="text-[10px] opacity-40 uppercase tracking-[0.2em] font-medium">Powered by Gemini & Postgres Vector Storage</p>
          </div>
        </div>
      </main>
    </div>
  );
}
