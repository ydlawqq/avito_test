"""Retriever для candidate generation: BM25 (rank_bm25).

Используемые open-source библиотеки:
  * rank_bm25 (BM25Okapi)        — лексический поиск
  * pymorphy3                    — лемматизация (см. src/data/preprocessing.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str  # "bm25"
    metadata: dict | None = None



class BM25Retriever:
    """BM25-индекс в памяти поверх предобработанных (лемматизированных) токенов.

    Корпус подаётся уже токенизированным: лемматизация 189k документов —
    дорогая операция, она выполняется один раз на этапе prepare_data.
    """

    def __init__(
        self,
        doc_ids: list[str],
        tokenized_docs: list[list[str]],
        texts: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ):
        self.doc_ids = list(doc_ids)
        self.corpus = [
            {
                "id": d,
                "text": (texts[i] if texts else " ".join(tokenized_docs[i])),
                "metadata": (metadatas[i] if metadatas else None),
            }
            for i, d in enumerate(doc_ids)
        ]
        self.bm25 = BM25Okapi(tokenized_docs)

    @classmethod
    def from_documents(cls, corpus: list[dict]) -> "BM25Retriever":
        """Совместимость с исходной заготовкой: корпус из dict с ключами id/text."""
        return cls(
            doc_ids=[d["id"] for d in corpus],
            tokenized_docs=[d["text"].lower().split() for d in corpus],
        )

    # -- основной API -------------------------------------------------------

    def score_tokens(self, tokens: list[str]) -> np.ndarray:
        """Скор BM25 для всех документов корпуса по токенам запроса."""
        return self.bm25.get_scores(tokens)

    def top_indices(self, tokens: list[str], top_k: int = 50) -> list[int]:
        """Индексы top-k документов по BM25 (score > 0)."""
        scores = self.score_tokens(tokens)
        # kind="stable": при равных скорах порядок задается индексом корпуса —
        # результат не зависит от версии numpy (гарантия детерминизма).
        top = np.argsort(-scores, kind="stable")[:top_k]
        return [int(i) for i in top if scores[i] > 0]

    def search_tokens(
        self,
        tokens: list[str],
        top_k: int = 50,
    ) -> list[RetrievedDoc]:
        """Поиск по предобработанному (лемматизированному) запросу."""
        idx = self.top_indices(tokens, top_k)
        scores = self.bm25.get_scores(tokens)
        return [
            RetrievedDoc(
                doc_id=self.doc_ids[i],
                text=self.corpus[i]["text"],
                score=float(scores[i]),
                source="bm25",
                metadata=self.corpus[i]["metadata"],
            )
            for i in idx
        ]

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        """Поиск по сырому тексту запроса (простая токенизация без лемматизации).

        В основном пайплайне используйте search_tokens — с лемматизацией метрика выше.
        """
        return self.search_tokens(query.lower().split(), top_k=top_k)
