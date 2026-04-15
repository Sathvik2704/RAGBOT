-- SQL Schema for RAG Chatbot Pro
-- Run this entire script in your Supabase SQL Editor

-- 1. Create the `documents` table
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    file_name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    chunks_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Configure Row Level Security (RLS) for anonymous access
-- Since we removed OAuth, we need to allow the 'anon' role to read, write, and delete
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow anonymous select" ON public.documents FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anonymous insert" ON public.documents FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "Allow anonymous delete" ON public.documents FOR DELETE TO anon USING (true);

-- 3. Create the Storage Bucket for file uploads
INSERT INTO storage.buckets (id, name, public) 
VALUES ('user-documents', 'user-documents', true)
ON CONFLICT (id) DO NOTHING;

-- 4. Create proper storage policies for anonymous access to the new bucket
CREATE POLICY "Allow anonymous read user-documents" ON storage.objects FOR SELECT TO anon USING (bucket_id = 'user-documents');
CREATE POLICY "Allow anonymous insert user-documents" ON storage.objects FOR INSERT TO anon WITH CHECK (bucket_id = 'user-documents');
CREATE POLICY "Allow anonymous delete user-documents" ON storage.objects FOR DELETE TO anon USING (bucket_id = 'user-documents');
