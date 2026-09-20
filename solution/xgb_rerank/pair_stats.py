"""Построение таблиц «парной статистики» для XGBoost-реранкера.

Источник сигналов — train.parquet, ТОЛЬКО позитивы train-split сессий
(первые 70% md5-порядка build_validation, тот же сплит, что в
artifacts/xgb_rerank/dataset.parquet). Сессии eval/holdout и benchmark-запросы
в статистику не попадают -> утечки нет, а на инференсе таблицы применимы
ко всем запросам (все search_location_id benchmark есть в train).

Таблицы (artifacts/xgb_rerank/pair_stats/):
    item_geo.parquet       item_id, lat, lon — координаты корпуса
    loc_centroid.parquet   location_id, lat, lon, n, source — центроиды локаций
                           (source=corpus: центроид items корпуса с этой локацией;
                            source=train: fallback — центроид позитивов train)
    loc_pairs.parquet      search_loc, item_loc, cnt, cnt_s, prob  (P(I|S))
    loc_microcat.parquet   search_loc, microcat_id, cnt_s, freq   P(microcat|S)
    microcat_pos.parquet   microcat_id, freq                      P(microcat)
    qtext_cnt.parquet      qtext, cnt_t
    qtext_microcat.parquet qtext, microcat_id, cnt_t, freq        P(microcat|text)
    qtext_item.parquet     qtext, item_id, cnt                    count(text, item)
    item_pos.parquet       item_id, cnt                           count(item)

Запуск: python solution/xgb_rerank/pair_stats.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import numpy as np
import pandas as pd

from src.evaluation.validation import QUERY_KEY_COLUMNS, build_validation


def stable_key(row) -> str:
    raw = "\x1f".join(str(v) for v in row)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Построение таблиц парной статистики для XGBoost-реранкера")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "artifacts" / "xgb_rerank" / "pair_stats"),
        help="каталог для таблиц (по умолчанию pair_stats; для purge-версии "
             "укажите отдельный каталог, чтобы не затирать рабочие таблицы)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ сессии
    sessions = build_validation(
        PROJECT_ROOT / "raw_data/train.parquet",
        PROJECT_ROOT / "artifacts/items_processed.parquet",
        size=10**9,
    )
    n_train = int(len(sessions) * 0.7)
    key2qid = {
        stable_key(vals): qid
        for vals, qid in zip(
            sessions[QUERY_KEY_COLUMNS].itertuples(index=False, name=None),
            sessions["query_id"],
        )
    }
    train_qids = set(sessions["query_id"].iloc[:n_train])
    eval_qids = set(sessions["query_id"].iloc[n_train:])
    corpus_ids = set(pd.read_parquet(
        PROJECT_ROOT / "artifacts/items_processed.parquet", columns=["item_id"]
    )["item_id"])
    print(f"сессий: {len(sessions)}, train-split: {len(train_qids)}")

    tr = pd.read_parquet(PROJECT_ROOT / "raw_data/train.parquet")
    tr["lat"] = pd.to_numeric(tr["item_latitude"], errors="coerce")
    tr["lon"] = pd.to_numeric(tr["item_longitude"], errors="coerce")
    tr["_key"] = [
        stable_key(vals)
        for vals in tr[QUERY_KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    tr["qid"] = tr["_key"].map(key2qid)
    # ------------------------------------------------- строки для статистик
    # СТРОГО только строки train-split сессий (первые 70% build_validation).
    # Раньше исключали лишь корпусные позитивы eval/holdout-сессий — из-за
    # этого в статистиках оставались клики train-сессий (self-leak: у 100%
    # train-позитивов item_pos_cnt_log>0) и некорпусные строки eval, а
    # распределение статистических фич на benchmark резко сдвинуто.
    is_train = tr["qid"].isin(train_qids)
    pos = tr[is_train]
    print(f"строк для статистик: {len(pos)} "
          f"(исключено {int((~is_train).sum())} строк вне train-split)")


    cnt_s = pos.groupby("search_location_id").size()

    # ------------------------------------------------------------- гео
    it = pd.read_parquet(
        PROJECT_ROOT / "raw_data/benchmark_items.parquet",
        columns=["item_id", "item_location_id", "item_latitude", "item_longitude"],
    )
    it["lat"] = pd.to_numeric(it["item_latitude"], errors="coerce")
    it["lon"] = pd.to_numeric(it["item_longitude"], errors="coerce")
    it[["item_id", "lat", "lon"]].to_parquet(out_dir / "item_geo.parquet", index=False)

    cent = (
        it.groupby("item_location_id")
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"))
        .reset_index()
    )
    cent["source"] = "corpus"
    # fallback: центроид позитивов train для локаций, которых нет в корпусе
    missing = set(cnt_s.index) - set(cent["item_location_id"])
    if missing:
        fb = (
            pos[pos["search_location_id"].isin(missing)]
            .groupby("search_location_id")
            .agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"))
            .reset_index()
            .rename(columns={"search_location_id": "item_location_id"})
        )
        fb["source"] = "train"
        cent = pd.concat([cent, fb], ignore_index=True)
    cent.to_parquet(out_dir / "loc_centroid.parquet", index=False)
    print(f"центроидов: {len(cent)} (fallback train: {(cent.source == 'train').sum()})")

    # -------------------------------------------------- локационные пары

    lp = pos.groupby(["search_location_id", "item_location_id"]).size().rename("cnt").reset_index()
    lp["cnt_s"] = lp["search_location_id"].map(cnt_s)
    lp["prob"] = lp["cnt"] / lp["cnt_s"]
    lp.to_parquet(out_dir / "loc_pairs.parquet", index=False)
    print(f"локационных пар (S,I): {len(lp)}")

    # ------------------------------------------------------- microcat-статы
    mc = pos.groupby("item_microcat_id").size()
    (mc / len(pos)).rename("freq").reset_index().to_parquet(
        out_dir / "microcat_pos.parquet", index=False
    )
    lm = (
        pos.groupby(["search_location_id", "item_microcat_id"]).size().rename("cnt").reset_index()
    )
    lm["cnt_s"] = lm["search_location_id"].map(cnt_s)
    lm["freq"] = lm["cnt"] / lm["cnt_s"]
    lm.to_parquet(out_dir / "loc_microcat.parquet", index=False)
    print(f"(S, microcat): {len(lm)}, microcat: {len(mc)}")

    # ------------------------------------------------------- текст запроса
    pos = pos.assign(qtext=pos["search_query"].astype(str))
    cnt_t = pos.groupby("qtext").size()
    cnt_t.rename("cnt_t").reset_index().to_parquet(out_dir / "qtext_cnt.parquet", index=False)
    qm = pos.groupby(["qtext", "item_microcat_id"]).size().rename("cnt").reset_index()
    qm["cnt_t"] = qm["qtext"].map(cnt_t)
    qm["freq"] = qm["cnt"] / qm["cnt_t"]
    qm.to_parquet(out_dir / "qtext_microcat.parquet", index=False)
    qi = pos.groupby(["qtext", "item_id"]).size().rename("cnt").reset_index()
    qi.to_parquet(out_dir / "qtext_item.parquet", index=False)
    ip = pos.groupby("item_id").size().rename("cnt")
    ip.reset_index().to_parquet(out_dir / "item_pos.parquet", index=False)
    print(
        f"qtext: {len(cnt_t)}, (text,microcat): {len(qm)}, "
        f"(text,item): {len(qi)}, items: {len(ip)}"
    )
    print("Готово:", out_dir)


if __name__ == "__main__":
    main()
