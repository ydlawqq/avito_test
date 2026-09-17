"""Класс BM25-ретривера из условия + dataclass RetrievedDoc.

Русские стоп-слова (nltk.corpus.stopwords) отфильтровываются и из документов
корпуса, и из запросов — симметрично, на этапе токенизации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import nltk
import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from rank_bm25 import BM25Okapi

try:
    RUSSIAN_STOPWORDS: frozenset[str] = frozenset(nltk_stopwords.words("russian"))
except LookupError:  # словарь stopwords ещё не скачан
    nltk.download("stopwords", quiet=True)
    RUSSIAN_STOPWORDS = frozenset(nltk_stopwords.words("russian"))


def _tokenize(text: str) -> list[str]:
    """Токенизация: lower-case + удаление русских стоп-слов."""
    if not text:
        return []
    return [t for t in text.lower().split() if t not in RUSSIAN_STOPWORDS]


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] | None = None


class BM25Retriever:
    """Lightweight BM25 index in memory. For production with >1M docs, swap for Elasticsearch."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_ids = [d["id"] for d in corpus]
        tokenized = [_tokenize(d["text"]) for d in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedDoc(
                doc_id=self.doc_ids[i],
                text=self.corpus[i]["text"],
                score=float(scores[i]),
                source="bm25",
                metadata=self.corpus[i].get("metadata"),
            )
            for i in top_indices
        ]