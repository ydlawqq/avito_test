"""Быстрый BM25-скоринг, математически идентичный BM25Okapi (rank_bm25).

rank_bm25.get_scores на КАЖДЫЙ терм запроса строит q_freq полным проходом
по 189k словарей — это узкое место грида. Здесь TF-матрица корпуса один раз
собирается в scipy.sparse CSC, а скор запроса считается по sparse-срезам
колонок: идентичная формула BM25Okapi (тот же idf-словарь, doc_len, avgdl),
но только по ненулевым документам терма.

Паритет с rank_bm25 проверяется тестом ниже (max|diff| ~ 1e-12).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class FastBM25:
    """Строится из натренированного BM25Okapi (retriever.bm25).

    k1/b НЕ фиксируются: scores(tokens, k1, b) — можно гридовать без пересбора.
    """

    def __init__(self, bm25) -> None:
        doc_freqs = bm25.doc_freqs          # list[dict term->tf]
        self.n_docs = len(doc_freqs)
        self.doc_len = np.asarray(bm25.doc_len, dtype=np.float64)
        self.avgdl = float(bm25.avgdl)
        self.idf = dict(bm25.idf)           # уже с epsilon-флором BM25Okapi

        # vocabulary: term -> column index
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for j, freqs in enumerate(doc_freqs):
            for term, tf in freqs.items():
                c = vocab.setdefault(term, len(vocab))
                rows.append(j)
                cols.append(c)
                data.append(tf)
        self.tf = sp.csc_matrix(
            (data, (rows, cols)), shape=(self.n_docs, len(vocab)), dtype=np.float64
        )
        self.vocab = vocab

    def scores(self, tokens: list[str], k1: float, b: float) -> np.ndarray:
        norm = k1 * (1.0 - b + b * self.doc_len / self.avgdl)  # (n_docs,)
        out = np.zeros(self.n_docs, dtype=np.float64)
        # ВАЖНО: без дедупликации — rank_bm25 умножает вклад терма на его
        # частоту в запросе (q_freq), повторные токены обязаны суммироваться.
        for t in tokens:
            col_idx = self.vocab.get(t)
            if col_idx is None:
                continue
            col = self.tf.getcol(col_idx)
            tf = col.data
            docs = col.indices
            out[docs] += self.idf[t] * (k1 + 1.0) * tf / (tf + norm[docs])
        return out


def parity_check(fast: FastBM25, bm25, tokens_per_query: list[list[str]], k1: float, b: float) -> float:
    """max |fast - rank_bm25| по списку запросов (после установки k1/b)."""
    bm25.k1 = float(k1)
    bm25.b = float(b)
    worst = 0.0
    for tokens in tokens_per_query:
        ref = bm25.get_scores(tokens)
        got = fast.scores(tokens, k1, b)
        worst = max(worst, float(np.abs(ref - got).max()))
    return worst
