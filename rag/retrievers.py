"""Three retrievers with the same interface, so they can be compared on the same evaluation set.

- TfidfRetriever: lexical baseline (scikit-learn), runs anywhere.
- DenseRetriever: Sentence Transformers embeddings + FAISS inner-product search (cosine on normalized vectors).
- FilteredRetriever: parses structured filters from the question, keeps only matching tickets,
  then ranks those with an inner retriever. This is "hybrid" search: metadata filter + similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.data import field_values
from rag.query_parser import parse_filters


@dataclass
class Hit:
    ticket_id: int
    score: float
    row: int  # position in the dataframe


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(df["rag_text"])

    def scores(self, question: str) -> np.ndarray:
        q = self.vectorizer.transform([question])
        return (self.matrix @ q.T).toarray().ravel()

    def search(self, question: str, k: int = 5, candidates: np.ndarray | None = None) -> list[Hit]:
        return _top_k(self.df, self.scores(question), k, candidates)


class DenseRetriever:
    name = "dense-minilm"

    def __init__(self, df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import faiss  # noqa: PLC0415 - optional heavy dependency
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.df = df
        self.model = SentenceTransformer(model_name)
        emb = self.model.encode(
            df["rag_text"].tolist(), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        ).astype("float32")
        self.embeddings = emb
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def scores(self, question: str) -> np.ndarray:
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        return (self.embeddings @ q.T).ravel()

    def search(self, question: str, k: int = 5, candidates: np.ndarray | None = None) -> list[Hit]:
        if candidates is None:
            q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            scores, idx = self.index.search(q, k)
            return [Hit(int(self.df.at[i, "ticket_id"]), float(s), int(i)) for s, i in zip(scores[0], idx[0])]
        return _top_k(self.df, self.scores(question), k, candidates)


class FilteredRetriever:
    """Apply filters parsed from the question, then rank the remaining tickets by similarity."""

    def __init__(self, df: pd.DataFrame, inner, parser=None, parser_name: str = "rules"):
        """`parser(question, values) -> {field: value}`; defaults to the rule-based parser."""
        self.df = df
        self.inner = inner
        self.values = field_values(df)
        self.parser = parser or parse_filters
        self.name = f"filtered({parser_name})+{inner.name}"

    def search(self, question: str, k: int = 5) -> list[Hit]:
        filters = self.parser(question, self.values)
        if not filters:
            return self.inner.search(question, k)
        mask = np.ones(len(self.df), dtype=bool)
        for field, value in filters.items():
            mask &= self.df[field].astype(str).to_numpy() == value
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            return []  # nothing matches: better to return nothing than unrelated tickets
        return self.inner.search(question, k, candidates=candidates)


def _top_k(df: pd.DataFrame, scores: np.ndarray, k: int, candidates: np.ndarray | None) -> list[Hit]:
    if candidates is not None:
        masked = np.full_like(scores, -np.inf, dtype=float)
        masked[candidates] = scores[candidates]
        scores = masked
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [Hit(int(df.at[i, "ticket_id"]), float(scores[i]), int(i)) for i in top]
