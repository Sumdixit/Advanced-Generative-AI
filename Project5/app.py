#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Powered HR Assistant (Nestlé HR Policy, 2012)
- Loads provided PDFs
- Creates a Chroma vector store with OpenAI embeddings
- Answers questions via Retrieval-Augmented Generation (RAG)
- Simple Gradio chat UI with sources

Usage:
  1) Ensure packages are installed:
       pip install -r requirements.txt
  2) Set your OpenAI API key:
       export OPENAI_API_KEY="sk-..."
  3) Run the app:
       python app.py
  4) Open the displayed Gradio URL.
"""

import os
import re
import json
import time
import hashlib
import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import gradio as gr
from openai import OpenAI

# ------------------------ Configuration ------------------------

# You can switch to "gpt-3.5-turbo" to match the assignment's wording.
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

DOC_PATHS = [
    "/mnt/data/the_nestle_hr_policy_pdf_2012.pdf",         # main HR policy (ground truth)
    "/mnt/data/Course_End_Project_Crafting_an_AI_Powered_HR_Assistant.pdf",  # project brief (meta)
    "/mnt/data/Gradio_Documentation.pdf",                  # gradio reference (meta)
]

# Where to store the Chroma DB (created/updated on index build)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_hr_policy_db")

# Chunking parameters
CHUNK_SIZE = 900           # characters
CHUNK_OVERLAP = 150        # characters

# Prompt guardrails
SYSTEM_PROMPT = """\
You are an HR policy assistant specialized in Nestlé's Human Resources Policy (2012).
Answer ONLY from the provided documents. If you don't find an answer, say you don't know.
Be precise, concise, and include brief citations with filename and page numbers when relevant.
"""

ANSWER_PROMPT_TEMPLATE = """\
You are answering a user question about Nestlé HR policy using retrieved context.
- Use ONLY facts from the context.
- If the question asks about UI or implementation details, you may reference the Gradio documentation context.

Question:
{question}

Retrieved context:
{context}

Now write the best possible answer. Cite sources inline like [source: filename.pdf p.X].
If unsure or not found in context, state that the document does not specify.
"""

# ------------------------ Utilities ------------------------

def _require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it via environment variable before running the app."
        )

def load_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) for the given PDF, 1-indexed pages."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # normalize whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        pages.append((i, txt))
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunker with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@dataclass
class ChunkMeta:
    source: str
    page: int
    chunk_id: str

def build_corpus(paths: List[str]) -> Tuple[List[str], List[ChunkMeta]]:
    """Load PDFs, split to chunks, create texts and metadata."""
    texts: List[str] = []
    metas: List[ChunkMeta] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        pages = load_pdf(path)
        for page_num, page_text in pages:
            if not page_text:
                continue
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                cid = hashlib.sha1(f"{path}-{page_num}-{idx}-{len(ch)}".encode()).hexdigest()
                texts.append(ch)
                metas.append(ChunkMeta(source=os.path.basename(path), page=page_num, chunk_id=cid))
    return texts, metas

def embed_and_index(texts: List[str], metas: List[ChunkMeta], persist_dir: str):
    """Create a Chroma collection with OpenAI embeddings and persist it."""
    _require_api_key()

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    # Delete collection if exists (fresh rebuild)
    try:
        client.delete_collection("nestle_hr_policy")
    except Exception:
        pass

    # Use Chroma's OpenAIEmbeddingFunction to keep dependencies light
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=DEFAULT_EMBED_MODEL
    )
    col = client.create_collection(name="nestle_hr_policy", embedding_function=ef)

    ids = [m.chunk_id for m in metas]
    metadatas = [{"source": m.source, "page": m.page} for m in metas]
    col.add(documents=texts, ids=ids, metadatas=metadatas)
    return col

def get_collection(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    # When reopening, must pass embedding function again
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=DEFAULT_EMBED_MODEL
    )
    try:
        return client.get_collection(name="nestle_hr_policy", embedding_function=ef)
    except Exception:
        return None

def retrieve(query: str, k: int = 5):
    col = get_collection(CHROMA_DIR)
    if col is None:
        raise RuntimeError("Index not found. Please click 'Build/Refresh Index' first.")
    res = col.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas

def render_context_with_citations(docs: List[str], metas: List[Dict]) -> str:
    """Build a context string and embed simple inline citations at the end of each chunk."""
    blocks = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        pg = m.get("page", "?")
        blocks.append(f"[{src} p.{pg}] {d}")
    return "\n\n".join(blocks)

def answer_with_gpt(question: str, context: str) -> str:
    _require_api_key()
    client = OpenAI()

    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question.strip(), context=context.strip())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()

# ------------------------ Gradio App ------------------------

with gr.Blocks(title="Nestlé HR Assistant (RAG)") as demo:
    gr.Markdown(
        """
        # Nestlé HR Assistant (RAG)
        Ask questions about the **Nestlé Human Resources Policy (2012)**.  
        - Click **Build/Refresh Index** once (first use) to parse PDFs and create the vector DB.  
        - Provide your OpenAI API key in the textbox (kept only in session memory).  
        - Your questions are answered **only** from the documents; answers include citations.
        """
    )
    with gr.Row():
        openai_key = gr.Textbox(label="OpenAI API Key (not stored)", type="password")
        build_btn = gr.Button("🔄 Build/Refresh Index")
    status = gr.Markdown("Index status: _unknown_")

    with gr.Row():
        chat = gr.Chatbot(type="messages", label="Chat")
    with gr.Row():
        question = gr.Textbox(label="Your question", placeholder="e.g., What are the key elements of Nestlé's Total Rewards?")
        ask_btn = gr.Button("▶️ Ask")

    sources_out = gr.HTML(label="Retrieved Sources")

    def do_build(key: str):
        if key:
            os.environ["OPENAI_API_KEY"] = key
        _require_api_key()
        texts, metas = build_corpus(DOC_PATHS)
        if not texts:
            return "⚠️ No text extracted from PDFs. Check file paths.", ""
        embed_and_index(texts, metas, CHROMA_DIR)
        return f"✅ Index built with {len(texts)} chunks.", ""

    def do_ask(history, key: str, q: str):
        if key:
            os.environ["OPENAI_API_KEY"] = key
        _require_api_key()
        docs, metas = retrieve(q, k=5)
        context = render_context_with_citations(docs, metas)
        ans = answer_with_gpt(q, context)
        # Build a small HTML of the top sources
        srcs = []
        for m in metas:
            srcs.append(f"{m.get('source')} p.{m.get('page')}")
        src_html = "<br/>".join(dict.fromkeys(srcs))  # unique while preserving order
        history = (history or []) + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        return history, f"<b>Top sources:</b><br/>{src_html}"

    build_btn.click(fn=do_build, inputs=[openai_key], outputs=[status, sources_out])
    ask_btn.click(fn=do_ask, inputs=[chat, openai_key, question], outputs=[chat, sources_out])

if __name__ == "__main__":
    demo.launch()
