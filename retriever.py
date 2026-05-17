"""
Lightweight Retrieval Engine: BM25 (Sparse) only.

Designed for Render free tier (512MB RAM limit).
No PyTorch, no FAISS, no sentence-transformers.

The index is loaded ONCE at FastAPI startup, not per request.
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple

from rank_bm25 import BM25Okapi


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# ──────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────

_bm25_index: Optional[BM25Okapi] = None
_catalog: List[Dict] = []
_rich_texts: List[str] = []


# Test type labels for rich text
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "K": "Knowledge & Skills",
    "B": "Biodata & Situational Judgment",
    "C": "Competency",
    "S": "Simulations",
    "D": "Development & 360",
}


def _build_rich_text(item: dict) -> str:
    """
    Build a rich text string for BM25 indexing.
    Combines name, test type label, description, and job levels.
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
# Index Loading
# ──────────────────────────────────────────────

def load_index() -> None:
    """
    Load catalog and build BM25 index.
    Called once at FastAPI startup.
    """
    global _bm25_index, _catalog, _rich_texts
    
    print("[INIT] Loading retrieval engine...")
    
    # Load catalog
    print(f"  Loading catalog from {CATALOG_PATH}...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)
    
    # Build rich text for each item
    _rich_texts = [_build_rich_text(item) for item in _catalog]
    
    # Build BM25 index
    tokenized_texts = [_tokenize(text) for text in _rich_texts]
    _bm25_index = BM25Okapi(tokenized_texts)
    
    print(f"  [OK] Loaded {len(_catalog)} assessments, BM25 ready")


def build_index(catalog_path: str = CATALOG_PATH) -> None:
    """Alias for load_index (backward compatibility)."""
    load_index()


def get_catalog() -> List[Dict]:
    """Return the full catalog (for /catalog endpoint)."""
    return _catalog


def search_by_keyword(query: str, limit: int = 20) -> List[Dict]:
    """
    Simple keyword search over catalog names and descriptions.
    No LLM needed — pure string matching.
    """
    query_lower = query.lower().strip()
    results = []
    
    for item in _catalog:
        name = item.get("name", "").lower()
        desc = item.get("description", "").lower()
        test_type = item.get("test_type", "").lower()
        
        if (query_lower in name or 
            query_lower in desc or
            query_lower in test_type):
            results.append(item)
    
    return results[:limit]


# ──────────────────────────────────────────────
# Search (BM25 only — lightweight)
# ──────────────────────────────────────────────

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
        if scores[idx] > 0:
            results.append((int(idx), float(scores[idx])))
    return results


def search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Search using BM25 keyword matching.
    
    Lightweight alternative to hybrid FAISS+BM25 pipeline.
    Designed for deployment on memory-constrained environments.
    """
    bm25_results = _bm25_search(query, top_k=top_k * 3)
    
    if not bm25_results:
        return []
    
    # Normalize scores
    max_score = bm25_results[0][1] if bm25_results else 1.0
    
    results = []
    for doc_idx, score in bm25_results[:top_k]:
        if doc_idx < len(_catalog):
            item = _catalog[doc_idx].copy()
            item["relevance_score"] = round(score / max_score, 4) if max_score > 0 else 0
            results.append(item)
    
    return results


# ──────────────────────────────────────────────
# CLI: Test from command line
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading BM25 index from catalog.json...")
    load_index()
    
    test_queries = [
        "Java developer",
        "personality assessment for senior leadership",
        "contact center agents",
        "Excel Word admin",
    ]
    
    for q in test_queries:
        print(f"\n  Query: '{q}'")
        results = search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"    {i + 1}. {r['name']} (score: {r['relevance_score']})")
