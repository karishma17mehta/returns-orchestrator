"""Lexical retriever over pre-chunked brand policy documents.

Loads the chunk DataFrame produced by the policy-index notebook
(policy_index_baseline/df_chunks.pkl: chunk_id, text, source, brand, ...)
and scores chunks by token overlap with the query — no embedding model
needed at serve time, which keeps the service light. For higher recall,
swap in the FAISS index the same notebook builds; anything satisfying
`(query, brands) -> list[str]` plugs into PolicyComplianceAgent.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LexicalPolicyRetriever:
    def __init__(self, chunks: list[dict], top_k: int = 3):
        """chunks: dicts with at least 'text' and 'brand' keys."""
        self.chunks = chunks
        self.top_k = top_k
        self._doc_tokens = [Counter(_tokens(c["text"])) for c in chunks]
        # Document frequency for a simple TF-IDF weighting.
        self._df: Counter = Counter()
        for tok_counts in self._doc_tokens:
            self._df.update(tok_counts.keys())
        self._n_docs = max(len(chunks), 1)

    @classmethod
    def from_pickle(cls, path: str, top_k: int = 3) -> "LexicalPolicyRetriever":
        with open(path, "rb") as f:
            df = pickle.load(f)  # pandas DataFrame from the indexing notebook
        return cls(df[["text", "brand"]].to_dict("records"), top_k=top_k)

    def __call__(self, query: str, brands: list[str] | None = None) -> list[str]:
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []
        brand_set = {b.lower() for b in brands} if brands else None
        scored: list[tuple[float, str]] = []
        for chunk, tok_counts in zip(self.chunks, self._doc_tokens):
            if brand_set and str(chunk.get("brand", "")).lower() not in brand_set:
                continue
            score = sum(
                (1 + math.log(tok_counts[t])) * math.log(self._n_docs / self._df[t])
                for t in q_tokens
                if t in tok_counts and self._df[t] > 0
            )
            if score > 0:
                scored.append((score, chunk["text"]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [text for _score, text in scored[: self.top_k]]
