"""
Hybrid Retrieval Engine: FAISS (Dense) + BM25 (Sparse) + RRF (Fusion)

- FAISS: Semantic similarity via bge-small-en-v1.5 embeddings
- BM25: Keyword matching for exact product names/terms
- RRF: Reciprocal Rank Fusion to merge both ranked lists

The index is loaded ONCE at FastAPI startup, not per request.
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "catalog_index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "index.pkl")

# Test type labels for rich text embedding
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "K": "Knowledge & Skills",
    "B": "Biodata & Situational Judgment",
    "C": "Competency",
    "S": "Simulations",
    "D": "Development & 360",
}

# RRF constant (standard default)
RRF_K = 60


# ──────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.Index] = None
_bm25_index: Optional[BM25Okapi] = None
_catalog: List[Dict] = []
_rich_texts: List[str] = []


def _build_rich_text(item: dict) -> str:
    """
    Build a rich text string for embedding.
    Combines name, test type label, description, and job levels
    for maximum semantic coverage.
    """
    test_types = item.get("test_type", "K")
    type_labels = []
    for code in test_types.split(","):
        code = code.strip()
        type_labels.append(TEST_TYPE_LABELS.get(code, code))
    type_str = ", ".join(type_labels)
    
    job_levels = " ".join(item.get("job_levels", []))
    description = item.get("description", "")
    
    return f"{item['name']} | {type_str} | {description} | {job_levels}"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


# ──────────────────────────────────────────────
# Index Building
# ──────────────────────────────────────────────

def build_index(catalog_path: str = CATALOG_PATH) -> None:
    """
    Build FAISS + BM25 indexes from catalog.json and save to disk.
    Run this ONCE locally before deployment.
    """
    global _model, _faiss_index, _bm25_index, _catalog, _rich_texts
    
    print(f"[LOAD] Loading catalog from {catalog_path}...")
    with open(catalog_path, "r", encoding="utf-8") as f:
        _catalog = json.load(f)
    
    print(f"  Found {len(_catalog)} assessments")
    
    # Build rich text for each item
    _rich_texts = [_build_rich_text(item) for item in _catalog]
    
    # Load embedding model
    print(f"[MODEL] Loading embedding model: {EMBEDDING_MODEL}...")
    _model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings
    print("[EMBED] Generating embeddings...")
    embeddings = _model.encode(
        _rich_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings, dtype="float32")
    
    # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dimension)
    _faiss_index.add(embeddings)
    print(f"[FAISS] Index built: {_faiss_index.ntotal} vectors, {dimension}d")
    
    # Build BM25 index
    tokenized_texts = [_tokenize(text) for text in _rich_texts]
    _bm25_index = BM25Okapi(tokenized_texts)
    print("[BM25] Index built")
    
    # Save to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(_faiss_index, FAISS_INDEX_PATH)
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({
            "catalog": _catalog,
            "rich_texts": _rich_texts,
        }, f)
    
    print(f"[SAVE] Saved index to {INDEX_DIR}/")


# ──────────────────────────────────────────────
# Index Loading
# ──────────────────────────────────────────────

def load_index() -> None:
    """
    Load pre-built indexes from disk.
    Called once at FastAPI startup via @app.on_event("startup").
    """
    global _model, _faiss_index, _bm25_index, _catalog, _rich_texts
    
    print("[INIT] Loading retrieval engine...")
    
    # Load embedding model
    print(f"  Loading model: {EMBEDDING_MODEL}...")
    _model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load FAISS index
    if os.path.exists(FAISS_INDEX_PATH):
        print("  Loading pre-built FAISS index...")
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("  [WARN] No pre-built FAISS index found, building from catalog...")
        build_index()
        return
    
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
        _catalog = metadata["catalog"]
        _rich_texts = metadata["rich_texts"]
    
    # Rebuild BM25 (it's fast, not worth serializing)
    tokenized_texts = [_tokenize(text) for text in _rich_texts]
    _bm25_index = BM25Okapi(tokenized_texts)
    
    print(f"  [OK] Loaded {len(_catalog)} assessments, FAISS={_faiss_index.ntotal} vectors")


def get_catalog() -> List[Dict]:
    """Return the full catalog (for /catalog endpoint)."""
    return _catalog


def search_by_keyword(query: str, limit: int = 20) -> List[Dict]:
    """
    Simple keyword search over catalog names and descriptions.
    No LLM or embeddings — pure string matching.
    Used by the /catalog/search endpoint.
    """
    query_lower = query.lower().strip()
    results = []
    
    for item in _catalog:
        name = item.get("name", "").lower()
        desc = item.get("description", "").lower()
        test_type = item.get("test_type", "").lower()
        
        # Check if query matches name, description, or test type
        if (query_lower in name or 
            query_lower in desc or
            query_lower in test_type):
            results.append(item)
    
    return results[:limit]


# ──────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────

def _faiss_search(query: str, top_k: int = 20) -> List[Tuple[int, float]]:
    """
    Dense semantic search via FAISS.
    Returns list of (index, score) tuples.
    """
    if _model is None or _faiss_index is None:
        return []
    
    query_embedding = _model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    
    scores, indices = _faiss_index.search(query_embedding, min(top_k, _faiss_index.ntotal))
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:  # FAISS returns -1 for missing results
            results.append((int(idx), float(score)))
    return results


def _bm25_search(query: str, top_k: int = 20) -> List[Tuple[int, float]]:
    """
    Sparse keyword search via BM25.
    Returns list of (index, score) tuples.
    """
    if _bm25_index is None:
        return []
    
    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)
    
    # Get top-k indices by score
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include non-zero BM25 scores
            results.append((int(idx), float(scores[idx])))
    return results


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = RRF_K
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion to merge multiple ranked lists.
    
    Formula: rrf_score(doc) = Σ 1 / (k + rank)
    
    This ignores raw scores (which are on incompatible scales)
    and uses rank position only.
    """
    doc_scores: Dict[int, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked_list):
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = 0.0
            doc_scores[doc_idx] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Hybrid search: FAISS + BM25 + RRF fusion + semantic reranking.
    
    Pipeline:
    1. FAISS dense retrieval (top_k * 3 candidates)
    2. BM25 sparse retrieval (top_k * 3 candidates)
    3. RRF fusion to merge both lists
    4. Semantic reranking: re-score using query-to-name cosine similarity
    
    Returns top_k catalog items ranked by combined relevance.
    """
    # Run both retrievers with extra candidates for reranking
    faiss_results = _faiss_search(query, top_k=top_k * 3)
    bm25_results = _bm25_search(query, top_k=top_k * 3)
    
    # Fuse with RRF
    fused = _reciprocal_rank_fusion([faiss_results, bm25_results])
    
    # Map back to catalog items (take more than needed for reranking)
    candidates = []
    for doc_idx, rrf_score in fused[:top_k * 2]:
        if doc_idx < len(_catalog):
            item = _catalog[doc_idx].copy()
            item["rrf_score"] = rrf_score
            candidates.append(item)
    
    # Rerank: compute cosine similarity between query and each candidate name
    if _model is not None and candidates:
        reranked = _semantic_rerank(query, candidates, top_k)
        return reranked
    
    # Fallback: return RRF-ranked results without reranking
    for item in candidates:
        item["relevance_score"] = round(item.pop("rrf_score", 0), 4)
    return candidates[:top_k]


def _semantic_rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """
    Rerank candidates using cosine similarity between query and candidate names.
    Uses the same embedding model already loaded — no extra dependencies.
    
    Combines RRF score (0.3 weight) with semantic score (0.7 weight)
    for a final relevance score.
    """
    import numpy as np
    
    # Encode query
    query_emb = _model.encode([query], normalize_embeddings=True)
    
    # Encode candidate names (batch for efficiency)
    candidate_texts = [
        f"{item['name']} {item.get('description', '')[:100]}"
        for item in candidates
    ]
    candidate_embs = _model.encode(candidate_texts, normalize_embeddings=True)
    
    # Compute cosine similarities
    similarities = np.dot(candidate_embs, query_emb.T).flatten()
    
    # Normalize RRF scores to 0-1 range
    rrf_scores = np.array([item.get("rrf_score", 0) for item in candidates])
    if rrf_scores.max() > 0:
        rrf_normalized = rrf_scores / rrf_scores.max()
    else:
        rrf_normalized = rrf_scores
    
    # Combined score: 70% semantic + 30% RRF
    combined_scores = 0.7 * similarities + 0.3 * rrf_normalized
    
    # Sort by combined score
    ranked_indices = np.argsort(combined_scores)[::-1]
    
    results = []
    for idx in ranked_indices[:top_k]:
        item = candidates[idx].copy()
        item.pop("rrf_score", None)
        item["relevance_score"] = round(float(combined_scores[idx]), 4)
        results.append(item)
    
    return results


# (search_by_keyword is defined above at line 184)


# ──────────────────────────────────────────────
# CLI: Build index from command line
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Building FAISS + BM25 index from catalog.json...")
    build_index()
    
    # Quick test
    print("\n[TEST] Testing hybrid search...")
    load_index()
    
    test_queries = [
        "Java developer",
        "personality assessment for senior leadership",
        "contact center agents",
    ]
    
    for q in test_queries:
        print(f"\n  Query: '{q}'")
        results = search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"    {i + 1}. {r['name']} (score: {r['relevance_score']})")
