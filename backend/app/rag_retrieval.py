"""
FleetSense - RAG Retrieval Layer
====================================
Embeds the maintenance knowledge base using sentence-transformers, indexes
it with FAISS for fast semantic search, and retrieves relevant documents
for a given query.

Model and index are cached as module-level singletons, loaded once on
first use and reused across every subsequent call - not reloaded from
disk on every query, which was needlessly spiking memory.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_docs = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_index_and_docs():
    global _index, _docs
    if _index is None:
        _index = faiss.read_index(f"{DATA_DIR}/maintenance_faiss.index")
    if _docs is None:
        with open(f"{DATA_DIR}/maintenance_knowledge_base.json") as f:
            _docs = json.load(f)
    return _index, _docs


def load_docs():
    with open(f"{DATA_DIR}/maintenance_knowledge_base.json") as f:
        return json.load(f)


def build_index():
    docs = load_docs()
    model = _get_model()

    texts = [f"{d['title']}. {d['content']}" for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, f"{DATA_DIR}/maintenance_faiss.index")
    print(f"Built FAISS index: {index.ntotal} documents, dimension {dimension}")

    return index, docs, model


def search(query: str, top_k: int = 2):
    """
    Retrieves the top_k most relevant maintenance docs for a given query.
    Model and index are loaded once (cached), not on every call.
    """
    model = _get_model()
    index, docs = _get_index_and_docs()

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "title": docs[idx]["title"],
            "content": docs[idx]["content"],
            "relevance_score": float(score),
        })
    return results


if __name__ == "__main__":
    build_index()

    print("\n" + "=" * 60)
    print("TEST RETRIEVAL")
    print("=" * 60)
    test_query = "why is this engine flagged as high risk"
    results = search(test_query, top_k=2)
    for r in results:
        print(f"\n[{r['relevance_score']:.3f}] {r['title']}")
        print(f"  {r['content'][:150]}...")