import React, { useState, useRef, useEffect } from 'react';
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
  Image as ImageIcon,
  Music,
  Paperclip,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import {
  chunkText,
  generateEmbeddings,
  generateImageEmbedding,
  generateAudioEmbedding,
  findRelevantChunks,
  type DocumentChunk,
} from './lib/rag';
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
  modality: 'text' | 'image' | 'audio';
}

type ServiceStatus = 'checking' | 'connected' | 'disconnected';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const [ollamaStatus, setOllamaStatus] = useState<ServiceStatus>('checking');
  const [embedStatus, setEmbedStatus] = useState<ServiceStatus>('checking');

  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadUserData();
    checkServices();
  }, []);

  // Check health of local services
  const checkServices = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setOllamaStatus(data.services?.ollama === 'connected' ? 'connected' : 'disconnected');
        setEmbedStatus(data.services?.embedServer === 'connected' ? 'connected' : 'disconnected');
      }
    } catch {
      setOllamaStatus('disconnected');
      setEmbedStatus('disconnected');
    }
  };

  const loadUserData = async () => {
    try {
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
          storagePath: d.storage_path,
          modality: d.modality || 'text',
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

  // ---------------------------------------------------------------
  // Detect file modality from MIME type
  // ---------------------------------------------------------------
  function detectModality(mimeType: string): 'text' | 'image' | 'audio' {
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('audio/')) return 'audio';
    return 'text';
  }

  // ---------------------------------------------------------------
  // File Upload Handler (multimodal)
  // ---------------------------------------------------------------
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const modality = detectModality(file.type);
    
    try {
      let chunksWithEmbeddings: DocumentChunk[] = [];
      const fileName = file.name;

      if (modality === 'text') {
        // ---- TEXT / PDF flow ----
        setUploadProgress('Extracting text...');
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/extract', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Failed to extract text from file.');
        }
        
        const { text } = await response.json();
        
        setUploadProgress('Chunking and Embedding...');
        const chunks = chunkText(text, fileName);
        
        for (let i = 0; i < chunks.length; i++) {
          setUploadProgress(`Embedding chunk ${i + 1}/${chunks.length}...`);
          const embedding = await generateEmbeddings(chunks[i].text);
          chunksWithEmbeddings.push({ ...chunks[i], embedding });
        }

      } else if (modality === 'image') {
        // ---- IMAGE flow ----
        setUploadProgress('Generating image embedding...');
        const embedding = await generateImageEmbedding(file);
        chunksWithEmbeddings = [{
          id: crypto.randomUUID(),
          text: `[Image: ${fileName}]`,
          source: fileName,
          page: 1,
          embedding,
          modality: 'image',
        }];

      } else if (modality === 'audio') {
        // ---- AUDIO flow ----
        setUploadProgress('Generating audio embedding...');
        const embedding = await generateAudioEmbedding(file);
        chunksWithEmbeddings = [{
          id: crypto.randomUUID(),
          text: `[Audio: ${fileName}]`,
          source: fileName,
          page: 1,
          embedding,
          modality: 'audio',
        }];
      }

      // Store file in Supabase storage
      setUploadProgress('Storing file securely...');
      const fileExt = file.name.split('.').pop();
      const storagePath = `uploads/${crypto.randomUUID()}.${fileExt}`;
      
      const { error: uploadError } = await supabase.storage
        .from('user-documents')
        .upload(storagePath, file);

      if (uploadError) throw uploadError;

      // Save to database
      setUploadProgress('Saving to knowledge base...');
      
      const { data: insertedDoc, error: dbError } = await supabase
        .from('documents')
        .insert({
          file_name: fileName,
          storage_path: storagePath,
          chunks_data: chunksWithEmbeddings,
          modality: modality,
        })
        .select()
        .single();

      if (dbError) throw dbError;

      if (insertedDoc) {
        setDocuments(prev => [...prev, { 
          id: insertedDoc.id, 
          name: fileName, 
          chunks: chunksWithEmbeddings,
          storagePath,
          modality,
        }]);
      }
      
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Failed to process ${modality} file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeDocument = async (id: string, storagePath?: string) => {
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

  // ---------------------------------------------------------------
  // Chat via Ollama
  // ---------------------------------------------------------------
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
        relevantChunks = await findRelevantChunks(query, allChunks);
        context = relevantChunks.map(c => {
          const modalityLabel = c.modality ? ` (${c.modality})` : '';
          return `[Source: ${c.source}${modalityLabel}, Page: ${c.page}]\n${c.text}`;
        }).join('\n\n');
      }

      const historyContext = messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n');

      const prompt = `You are a helpful and accurate RAG chatbot. Use the provided context and conversation history to answer the user's query.

CONSTRAINTS:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say "I don't know. The provided documents do not contain information about this."
3. Be concise and professional.
4. Maintain a coherent conversation based on the history.
5. For image and audio sources, describe what was matched and why it's relevant.

CONVERSATION HISTORY:
${historyContext}

RETRIEVED CONTEXT:
${context || 'No documents uploaded yet.'}

USER QUERY:
${query}

ASSISTANT RESPONSE:`;

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(err.error || err.details || 'Chat request failed');
      }

      const data = await response.json();
      const responseText = data.response || "I'm sorry, I couldn't generate a response.";

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
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get a response from Ollama. Make sure Ollama is running.'}`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleFeedback = (messageId: string, feedback: 'up' | 'down') => {
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback } : m));
  };

  // ---------------------------------------------------------------
  // Icon helper for file modality
  // ---------------------------------------------------------------
  function ModalityIcon({ modality, className }: { modality: string; className?: string }) {
    switch (modality) {
      case 'image':
        return <ImageIcon className={cn("w-4 h-4", className)} />;
      case 'audio':
        return <Music className={cn("w-4 h-4", className)} />;
      default:
        return <FileText className={cn("w-4 h-4", className)} />;
    }
  }

  function StatusDot({ status }: { status: ServiceStatus }) {
    if (status === 'checking') {
      return <Loader2 className="w-3 h-3 animate-spin text-yellow-500" />;
    }
    if (status === 'connected') {
      return <CheckCircle2 className="w-3 h-3 text-green-500" />;
    }
    return <XCircle className="w-3 h-3 text-red-500" />;
  }

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
          <p className="text-[10px] opacity-60 uppercase tracking-[0.2em] mb-4">MULTIMODAL • LOCAL</p>

          {/* Service Status Indicators */}
          <div className="flex gap-4 mb-4 text-[10px] font-mono">
            <div className="flex items-center gap-1.5">
              <StatusDot status={ollamaStatus} />
              <span className="opacity-70">Ollama</span>
            </div>
            <div className="flex items-center gap-1.5">
              <StatusDot status={embedStatus} />
              <span className="opacity-70">Embeddings</span>
            </div>
          </div>
          
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 py-3 border border-[#141414] hover:bg-[#141414] hover:text-[#E4E3E0] transition-all duration-200 disabled:opacity-50"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="text-xs font-semibold uppercase tracking-wider">Upload File</span>
          </button>
          <p className="mt-2 text-[9px] text-center opacity-40 font-mono">PDF • TXT • JPG • PNG • WEBP • WAV • MP3 • OGG</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept=".pdf,.txt,.jpg,.jpeg,.png,.webp,.wav,.mp3,.ogg,.flac"
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
              <p className="text-[9px] opacity-30 mt-1">Upload PDFs, images, or audio files</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.id} className="group flex items-center justify-between p-3 border border-[#141414] bg-white/50 hover:bg-[#141414] hover:text-[#E4E3E0] transition-colors cursor-default">
                <div className="flex items-center gap-3 overflow-hidden">
                  <ModalityIcon modality={doc.modality} className="flex-shrink-0 opacity-70" />
                  <div className="overflow-hidden">
                    <span className="text-xs font-medium truncate block">{doc.name}</span>
                    <span className="text-[9px] opacity-50 uppercase tracking-wider">{doc.modality}</span>
                  </div>
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
          <div className="grid grid-cols-3 gap-2">
            <div className="p-2 border border-[#141414] bg-white/80">
              <div className="text-[10px] opacity-60">Messages</div>
              <div className="text-base font-serif italic">{messages.length}</div>
            </div>
            <div className="p-2 border border-[#141414] bg-white/80">
              <div className="text-[10px] opacity-60">Docs</div>
              <div className="text-base font-serif italic">{documents.length}</div>
            </div>
            <div className="p-2 border border-[#141414] bg-white/80">
              <div className="text-[10px] opacity-60">Types</div>
              <div className="text-[9px] font-mono mt-1">
                {documents.filter(d => d.modality === 'text').length}T {documents.filter(d => d.modality === 'image').length}I {documents.filter(d => d.modality === 'audio').length}A
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative bg-[#F4F4F4]">
        {/* Header */}
        <header className="h-16 border-b border-[#141414] flex items-center px-8 justify-between bg-white/50 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-2 h-2 rounded-full shadow-[0_0_8px_rgba(34,197,94,0.6)]",
              ollamaStatus === 'connected' ? "bg-green-500 animate-pulse" : "bg-red-500"
            )} />
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
              <p className="text-sm mb-4">Upload documents, images, or audio files to your knowledge base. Then ask questions — I'll find relevant content across all modalities.</p>
              <div className="flex gap-6 text-[10px] font-mono opacity-60">
                <div className="flex items-center gap-1.5">
                  <FileText className="w-3 h-3" /> PDF/TXT
                </div>
                <div className="flex items-center gap-1.5">
                  <ImageIcon className="w-3 h-3" /> Images
                </div>
                <div className="flex items-center gap-1.5">
                  <Music className="w-3 h-3" /> Audio
                </div>
              </div>
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
                            <div className="flex items-center gap-1.5">
                              <ModalityIcon modality={source.modality || 'text'} className="w-3 h-3" />
                              <span className="font-bold truncate max-w-[150px] text-blue-700">{source.source}</span>
                            </div>
                            <span className="opacity-50">{source.modality === 'text' ? `PG ${source.page}` : source.modality?.toUpperCase()}</span>
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
                <span className="text-xs font-mono italic opacity-70">Synthesizing response via Ollama...</span>
              </div>
            </motion.div>
          )}
        </div>

        {/* Input */}
        <div className="p-8 bg-[#E4E3E0] border-t border-[#141414] z-10">
          <div className="max-w-4xl mx-auto relative group">
            <div className="flex gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex-shrink-0 px-3 py-4 border border-[#141414] bg-white hover:bg-[#141414] hover:text-[#E4E3E0] transition-all disabled:opacity-30"
                title="Attach file (PDF, Image, Audio)"
              >
                <Paperclip className="w-4 h-4" />
              </button>
              <input 
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder={documents.length > 0 ? "Query the knowledge base..." : "Upload documents to begin..."}
                disabled={isThinking}
                className="flex-1 pl-6 pr-16 py-4 bg-white border border-[#141414] focus:outline-none focus:ring-2 focus:ring-blue-600/20 transition-all placeholder:italic placeholder:opacity-40 text-base shadow-sm"
              />
              <button 
                onClick={handleSend}
                disabled={!input.trim() || isThinking}
                className="flex-shrink-0 px-4 py-4 bg-[#141414] text-[#E4E3E0] hover:bg-blue-600 disabled:opacity-30 disabled:hover:bg-[#141414] transition-all flex items-center justify-center"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="mt-4 text-center">
            <p className="text-[10px] opacity-40 uppercase tracking-[0.2em] font-medium">Powered by Ollama & Local Multimodal Embeddings</p>
          </div>
        </div>
      </main>
    </div>
  );
}
