"""
Local RAG App — FastAPI backend
Exposes:
  POST /ingest        → scrape URL, embed, store in ChromaDB (streams progress via SSE)
  GET  /embeddings    → return stored embedding metadata for live viz
  GET  /inspect       → full vector DB viewer with pagination, search, and full vectors
  GET  /vector/{id}   → return the full 384-dim vector for a single chunk
  POST /similar       → find chunks most similar to a given chunk_id (cosine sim)
  POST /query         → embed query, retrieve top-k chunks, call Ollama LLM
  DELETE /reset       → clear the vector store
"""

import json
import re
import time
import uuid
from typing import Generator

import chromadb
import numpy as np
import ollama
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Init ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Local RAG App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast, local, 384-dim
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_store")
COLLECTION = CHROMA_CLIENT.get_or_create_collection("docs")

CHUNK_SIZE = 400   # characters per chunk
CHUNK_OVERLAP = 80
LLM_MODEL = "mistral"  # change to llama3, phi3, etc.


# ── Helpers ───────────────────────────────────────────────────────────────────
def scrape_url(url: str) -> str:
    """Fetch and strip HTML → plain text."""
    headers = {"User-Agent": "Mozilla/5.0 (RAG-Bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text: str) -> list[str]:
    """Sliding-window character chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def vec_stats(vec: list[float]) -> dict:
    """Compute useful statistics for a raw embedding vector."""
    v = np.array(vec, dtype=np.float32)
    return {
        "norm":          round(float(np.linalg.norm(v)), 6),
        "mean":          round(float(v.mean()), 6),
        "std":           round(float(v.std()), 6),
        "min":           round(float(v.min()), 6),
        "max":           round(float(v.max()), 6),
        "positive_dims": int((v > 0).sum()),
        "negative_dims": int((v < 0).sum()),
        "zero_dims":     int((v == 0).sum()),
        "sparsity":      round(float((v == 0).sum()) / len(v), 4),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str


def ingest_stream(url: str) -> Generator[str, None, None]:
    """SSE generator — yields progress + each embedding vector."""

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    yield sse({"type": "status", "msg": f"Fetching {url} …"})

    try:
        text = scrape_url(url)
    except Exception as e:
        yield sse({"type": "error", "msg": str(e)})
        return

    yield sse({"type": "status", "msg": f"Scraped {len(text):,} chars. Chunking …"})
    chunks = chunk_text(text)
    yield sse({"type": "status", "msg": f"Created {len(chunks)} chunks. Embedding …"})

    doc_id = str(uuid.uuid4())[:8]

    for i, chunk in enumerate(chunks):
        vec = EMBED_MODEL.encode(chunk).tolist()
        stats = vec_stats(vec)

        chunk_id = f"{doc_id}-{i}"
        COLLECTION.upsert(
            ids=[chunk_id],
            embeddings=[vec],
            documents=[chunk],
            metadatas=[{"url": url, "chunk": i, "total": len(chunks)}],
        )

        # Stream embedding preview (first 8 dims) for live viz
        yield sse({
            "type":         "embedding",
            "chunk_id":     chunk_id,
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "preview_dims": vec[:8],   # bar chart in left panel
            "norm":         stats["norm"],
            "snippet":      chunk[:120],
        })

        time.sleep(0.05)

    yield sse({"type": "done", "msg": f"Ingested {len(chunks)} chunks from {url}"})


@app.post("/ingest")
async def ingest(req: IngestRequest):
    return StreamingResponse(
        ingest_stream(req.url),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /embeddings  (header badge count) ────────────────────────────────────────
@app.get("/embeddings")
def get_embeddings():
    """Return stored chunk metadata for the UI badge (no raw vectors)."""
    results = COLLECTION.get(include=["metadatas", "documents"])
    items = []
    for i, doc_id in enumerate(results["ids"]):
        items.append({
            "id":      doc_id,
            "snippet": results["documents"][i][:100],
            "meta":    results["metadatas"][i],
        })
    return {"count": len(items), "items": items}


# ── /inspect  (full Vector DB viewer) ────────────────────────────────────────
@app.get("/inspect")
def inspect(page: int = 0, page_size: int = 20, search: str = ""):
    """
    Full vector DB viewer.
    Returns paginated chunks with full text, metadata, stats, and embedding preview.
    FIX: explicitly pass include=["embeddings"] so ChromaDB returns vectors.
    """
    # ✅ KEY FIX: include "embeddings" explicitly — ChromaDB omits them by default
    results = COLLECTION.get(
        include=["metadatas", "documents", "embeddings"]
    )

    ids        = results["ids"]
    docs       = results["documents"]
    metas      = results["metadatas"]
    embeddings = results.get("embeddings")  # may be numpy array or list
    if embeddings is None:
        embeddings = []

    if len(ids) == 0:
        return {
            "total": 0, "page": 0, "page_size": page_size,
            "pages": 1, "sources": [], "avg_norm": 0, "items": [],
        }

    # Convert embeddings to plain Python lists (ChromaDB may return numpy arrays)
    embeddings = [list(e) if e is not None else [] for e in embeddings]

    # Keyword filter
    if search:
        s = search.lower()
        filtered = [
            (i, d, m, e)
            for i, d, m, e in zip(ids, docs, metas, embeddings)
            if s in d.lower()
        ]
    else:
        filtered = list(zip(ids, docs, metas, embeddings))

    total      = len(filtered)
    page_items = filtered[page * page_size: (page + 1) * page_size]

    items = []
    for chunk_id, doc, meta, vec in page_items:
        vec = list(vec) if vec is not None else []
        stats = vec_stats(vec) if vec else {}

        # Histogram buckets: split 384 dims into 32 bins of 12 dims each,
        # store the mean absolute value per bin — used by the sparkline chart.
        histogram = []
        if vec:
            bin_size = max(1, len(vec) // 32)
            for b in range(0, len(vec), bin_size):
                bucket = vec[b: b + bin_size]
                histogram.append(round(float(np.mean(np.abs(bucket))), 5))

        items.append({
            "id":            chunk_id,
            "text":          doc,
            "url":           meta.get("url", ""),
            "chunk_index":   meta.get("chunk", 0),
            "total_chunks":  meta.get("total", 0),
            "dims":          len(vec),
            "preview_dims":  [float(x) for x in vec[:32]],   # first 32 dims → heatmap
            "histogram":     histogram,                        # 32 bins → sparkline
            # ✅ FIX: explicitly list stats fields instead of **stats unpacking
            "norm":          stats.get("norm", 0.0),
            "mean":          stats.get("mean", 0.0),
            "std":           stats.get("std", 0.0),
            "min":           stats.get("min", 0.0),
            "max":           stats.get("max", 0.0),
            "positive_dims": stats.get("positive_dims", 0),
            "negative_dims": stats.get("negative_dims", 0),
            "zero_dims":     stats.get("zero_dims", 0),
            "sparsity":      stats.get("sparsity", 0.0),
        })

    all_norms = [
        float(np.linalg.norm(np.array(list(e), dtype=np.float32)))
        for _, _, _, e in filtered if e is not None
    ]
    sources = list({m.get("url", "") for _, _, m, _ in filtered})

    return {
        "total":    total,
        "page":     page,
        "page_size": page_size,
        "pages":    max(1, -(-total // page_size)),
        "sources":  sources,
        "avg_norm": round(sum(all_norms) / len(all_norms), 4) if all_norms else 0,
        "items":    items,
    }


# ── /vector/{id}  (full 384-dim vector for one chunk) ────────────────────────
@app.get("/vector/{chunk_id}")
def get_vector(chunk_id: str):
    """Return the complete embedding vector for a single chunk."""
    result = COLLECTION.get(
        ids=[chunk_id],
        include=["embeddings", "documents", "metadatas"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Chunk not found")

    vec   = list(result["embeddings"][0])
    doc   = result["documents"][0]
    meta  = result["metadatas"][0]
    stats = vec_stats(vec)

    return {
        "id":            chunk_id,
        "dims":          len(vec),
        "vector":        [float(x) for x in vec],
        "text":          doc,
        "url":           meta.get("url", ""),
        "chunk_index":   meta.get("chunk", 0),
        "norm":          stats["norm"],
        "mean":          stats["mean"],
        "std":           stats["std"],
        "min":           stats["min"],
        "max":           stats["max"],
        "positive_dims": stats["positive_dims"],
        "negative_dims": stats["negative_dims"],
        "zero_dims":     stats["zero_dims"],
        "sparsity":      stats["sparsity"],
    }


# ── /similar  (find nearest neighbours by cosine similarity) ─────────────────
class SimilarRequest(BaseModel):
    chunk_id: str
    top_k: int = 5


@app.post("/similar")
def find_similar(req: SimilarRequest):
    """
    Given a chunk_id, fetch its vector from ChromaDB and return the
    top-k most similar chunks using cosine similarity.
    """
    # Fetch the query chunk's vector
    result = COLLECTION.get(
        ids=[req.chunk_id],
        include=["embeddings"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Chunk not found")

    query_vec = list(result["embeddings"][0])

    # Query ChromaDB for nearest neighbours
    results = COLLECTION.query(
        query_embeddings=[query_vec],
        n_results=min(req.top_k + 1, COLLECTION.count()),
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    items = []
    for i, (cid, doc, meta, dist, vec) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0],
    )):
        if cid == req.chunk_id:
            continue   # skip the query chunk itself
        cos_sim = cosine_similarity(query_vec, list(vec))
        items.append({
            "id":       cid,
            "snippet":  doc[:150],
            "url":      meta.get("url", ""),
            "distance": round(dist, 6),
            "similarity": round(cos_sim, 4),
        })
        if len(items) >= req.top_k:
            break

    return {"query_id": req.chunk_id, "results": items}


# ── /query  (RAG chat) ────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/query")
async def query(req: QueryRequest):
    """Embed question → retrieve → LLM → stream answer."""

    def generate():
        yield f"data: {json.dumps({'type': 'status', 'msg': 'Embedding your question …'})}\n\n"

        q_vec = EMBED_MODEL.encode(req.question).tolist()

        results = COLLECTION.query(
            query_embeddings=[q_vec],
            n_results=min(req.top_k, COLLECTION.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        distances = results["distances"][0]
        metas     = results["metadatas"][0]

        yield f"data: {json.dumps({'type': 'retrieved', 'chunks': [{'snippet': d[:120], 'score': round(1 - s, 3), 'url': m.get('url', '')} for d, s, m in zip(docs, distances, metas)]})}\n\n"

        context = "\n\n---\n\n".join(docs)
        prompt  = (
            f"You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {req.question}\n\n"
            f"Answer:"
        )

        yield f"data: {json.dumps({'type': 'status', 'msg': f'Calling {LLM_MODEL} via Ollama …'})}\n\n"

        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /reset ────────────────────────────────────────────────────────────────────
@app.delete("/reset")
def reset():
    """Clear the vector store."""
    CHROMA_CLIENT.delete_collection("docs")
    global COLLECTION
    COLLECTION = CHROMA_CLIENT.get_or_create_collection("docs")
    return {"msg": "Collection cleared"}
