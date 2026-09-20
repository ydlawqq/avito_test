"""Подготовка данных: лемматизация корпуса + построение локальной валидации.

Запуск:
    python scripts/prepare_data.py [--config configs/config.yaml]

Результат (в artifacts/):
    items_processed.parquet — корпус с лемматизированными полями:
        item_id, item_category_id, item_microcat_id, item_location_id,
        title_lem / params_lem / desc_lem  (list[str] лемм),
        title_raw / params_raw / desc_raw  (сырой текст объявлений)
    validation.parquet — локальная валидация из train.parquet
        (query_id, признаки запроса, relevant_items)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import pandas as pd
from tqdm import tqdm

from src.data.preprocessing import get_stopwords, lemmatize_many
from src.evaluation.validation import build_validation
from src.pipeline.common import artifacts_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = artifacts_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    desc_max = cfg["preprocessing"]["description_max_chars"]
    n_jobs = cfg["preprocessing"]["n_jobs"]
    stopwords = get_stopwords()
    print(f"Стоп-слов загружено: {len(stopwords)}; воркеров лемматизации: {n_jobs}")

    # ------------------------------------------------------------------ корпус
    items = pd.read_parquet(raw_dir / "benchmark_items.parquet")
    for col in ["item_title_raw", "item_infm_params_text", "item_description_raw"]:
        items[col] = items[col].fillna("").astype(str).str.strip()
    # Усекаем длинные описания — хвост почти не влияет на retrieval,
    # а лемматизация 189k документов без усечения заняла бы часы
    items["desc_capped"] = items["item_description_raw"].str.slice(0, desc_max)

    print("Лемматизация заголовков...")
    title_lem = lemmatize_many(items["item_title_raw"].tolist(), stopwords, n_jobs=n_jobs)
    print("Лемматизация параметров...")
    params_lem = lemmatize_many(
        items["item_infm_params_text"].tolist(), stopwords, n_jobs=n_jobs
    )
    print("Лемматизация описаний...")
    desc_lem = lemmatize_many(items["desc_capped"].tolist(), stopwords, n_jobs=n_jobs)

    processed = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "item_category_id": items["item_category_id"],
            "item_microcat_id": items["item_microcat_id"],
            "item_location_id": items["item_location_id"],
            "title_lem": title_lem,
            "params_lem": params_lem,
            "desc_lem": desc_lem,
            "title_raw": items["item_title_raw"],
            "params_raw": items["item_infm_params_text"],
            "desc_raw": items["desc_capped"],
        }
    )
    path = out_dir / "items_processed.parquet"
    processed.to_parquet(path, index=False)
    print(f"Сохранён корпус: {path} ({len(processed)} объявлений)")

    # -------------------------------------------------------------- валидация
    print("Построение локальной валидации из train.parquet...")
    validation = build_validation(
        train_path=raw_dir / "train.parquet",
        items_path=raw_dir / "benchmark_items.parquet",
        size=cfg["validation"]["size"],
        seed=cfg["validation"]["seed"],
    )
    path = out_dir / "validation.parquet"
    validation.to_parquet(path, index=False)
    rel_counts = validation["relevant_items"].apply(len)
    print(
        f"Сохранена валидация: {path} "
        f"({len(validation)} запросов, "
        f"релевантных на запрос: mean={rel_counts.mean():.2f}, max={rel_counts.max()})"
    )


if __name__ == "__main__":
    main()
