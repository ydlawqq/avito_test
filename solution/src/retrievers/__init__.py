"""Retriever: BM25 (candidate generation для XGB-реранкера)."""
from .retrieval import BM25Retriever, RetrievedDoc

__all__ = [
    "BM25Retriever",
    "RetrievedDoc",
]
