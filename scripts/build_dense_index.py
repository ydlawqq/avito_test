"""Построение dense-индекса (faiss) по чанкам объявлений.

Запуск (рекомендуется GPU-машина):
    python scripts/build_dense_index.py [--config configs/config.yaml]

Стратегия чанкования:
  * чанк №1: заголовок + параметры объявления (title + infm_params) —
    самый важный текст, всегда присутствует;
  * далее — скользящие окна по описанию (chunk_words слов с перекрытием
    chunk_overlap), максимум max_chunks_per_item чанков на объявление;
  * заголовок добавляется в начало каждого чанка описания — так чанк
    сохраняет контекст услуги.

Индекс: IndexIDMap2(FlatIP) по L2-нормализованным эмбеддингам —
точный inner product = косинусная близость. Корпус 189k чанков —
плоский индекс ищет быстро и не теряет полноту (важно для Recall@50).

Результат (в artifacts/dense/):
    e5.index          — faiss-индекс чанков
    chunk_item_ids.npy — маппинг internal_id чанка -> item_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.pipeline.common import artifacts_dir, load_config, load_items


def build_chunks(
    title: str,
    params: str,
    description: str,
    chunk_words: int,
    chunk_overlap: int,
    max_chunks: int,
) -> list[str]:
    """Разбить объявление на 1..max_chunks чанков текста."""
    head = ". ".join(x for x in [title.strip(), params.strip()] if x)

    words = description.split()
    if not words or max_chunks <= 1:
        return [head]

    step = max(1, chunk_words - chunk_overlap)
    chunks = [head]
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        # Заголовок дублируем в каждом чанке — контекст услуги
        chunks.append(f"{head}. {' '.join(window)}")
        if len(chunks) >= max_chunks:
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dense"]
    out_dir = artifacts_dir(cfg) / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(cfg)
    print(f"Корпус: {len(items)} объявлений")

    # ------------------------------------------------------------------ чанки
    chunk_texts: list[str] = []
    chunk_item_ids: list[str] = []
    for row in tqdm(items.itertuples(), total=len(items), desc="Чанкование"):
        chunks = build_chunks(
            row.title_raw,
            row.params_raw,
            row.desc_raw,
            chunk_words=dcfg["chunk_words"],
            chunk_overlap=dcfg["chunk_overlap"],
            max_chunks=dcfg["max_chunks_per_item"],
        )
        chunk_texts.extend(f"{dcfg['passage_prefix']}{c}" for c in chunks)
        chunk_item_ids.extend([row.item_id] * len(chunks))
    print(f"Всего чанков: {len(chunk_texts)}")

    # -------------------------------------------------------------- эмбеддинги
    print(f"Загрузка модели: {dcfg['model_name']}")
    model = SentenceTransformer(dcfg["model_name"])
    embeddings = model.encode(
        chunk_texts,
        batch_size=dcfg["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=True,
        max_length=dcfg["max_length"],
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    print(f"Эмбеддинги: {embeddings.shape}")

    # ------------------------------------------------------------------ индекс
    dim = embeddings.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    ids = np.arange(len(chunk_texts), dtype="int64")
    index.add_with_ids(embeddings, ids)

    index_path = out_dir / "baaim3.index"
    faiss.write_index(index, str(index_path))
    np.save(out_dir / "chunk_item_ids.npy", np.asarray(chunk_item_ids))
    print(f"Сохранён индекс: {index_path}; маппинг: chunk_item_ids.npy")


if __name__ == "__main__":
    main()
