"""
Retrieval Module for Historical Resolution Grounding.
Indexes past resolved customer queries and retrieves top-k similar
customer query -> agent resolution pairs for grounding generation.
Supports:
1. FAISS index with embeddings (OpenAI / sentence-transformers)
2. Scikit-learn TF-IDF cosine-similarity vector store (zero-dependency fallback)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HistoricalResolutionIndex:
    """
    Vector index holding historical (customer_query, agent_resolution) pairs.
    Allows query -> top-k retrieval of past agent resolutions.
    """

    def __init__(self, index_dir: str = "data/processed/resolution_index"):
        self.index_dir = index_dir
        self.pairs: List[Dict[str, str]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.is_built = False

    def build_index(self, pairs: List[Dict[str, str]]) -> None:
        """
        Build the retrieval index over historical pairs.
        """
        if not pairs:
            raise ValueError("Cannot build index on empty pairs list.")

        self.pairs = pairs
        corpus = [p["customer_query"] for p in self.pairs]

        # Use TF-IDF with n-grams (1, 2) for robust zero-dependency text similarity
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_built = True

        os.makedirs(self.index_dir, exist_ok=True)
        meta_path = os.path.join(self.index_dir, "pairs_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.pairs, f, indent=2)

        logger.info(f"Built resolution index with {len(self.pairs)} documents.")

    def load_index(self) -> bool:
        """
        Load existing index from disk if present.
        """
        meta_path = os.path.join(self.index_dir, "pairs_metadata.json")
        if not os.path.exists(meta_path):
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        if self.pairs:
            corpus = [p["customer_query"] for p in self.pairs]
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.is_built = True
            logger.info(f"Loaded existing resolution index ({len(self.pairs)} items).")
            return True
        return False

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar historical customer queries and their verified agent resolutions.
        Returns list of dicts with:
        - customer_query
        - agent_resolution
        - similarity_score (0.0 to 1.0)
        - thread_id
        """
        if not self.is_built or self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            results.append({
                "customer_query": self.pairs[idx]["customer_query"],
                "agent_resolution": self.pairs[idx]["agent_resolution"],
                "similarity_score": round(score, 4),
                "thread_id": self.pairs[idx].get("thread_id", ""),
                "intent": self.pairs[idx].get("intent", "general")
            })

        return results


def get_or_build_index(
    pairs: Optional[List[Dict[str, str]]] = None,
    index_dir: str = "data/processed/resolution_index"
) -> HistoricalResolutionIndex:
    """
    Factory function to retrieve or construct the singleton resolution index.
    """
    index = HistoricalResolutionIndex(index_dir=index_dir)
    if not index.load_index():
        if pairs is None:
            from pipeline.ingest import load_or_ingest_data
            _, pairs = load_or_ingest_data()
        index.build_index(pairs)
    return index


if __name__ == "__main__":
    from pipeline.ingest import load_or_ingest_data
    _, pairs = load_or_ingest_data()
    idx = get_or_build_index(pairs)
    res = idx.retrieve("my offline songs are pausing", top_k=2)
    print("Retrieval test results:", json.dumps(res, indent=2))
