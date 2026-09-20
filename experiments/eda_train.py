"""EDA train.parquet: связи признаков запроса (search_*) с признаками объявления.

Запуск: .venv/bin/python experiments/eda_train.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

tr = pd.read_parquet(ROOT / "raw_data/train.parquet")
it = pd.read_parquet(ROOT / "raw_data/benchmark_items.parquet")
bq = pd.read_parquet(ROOT / "raw_data/benchmark_queries.parquet")

print("=" * 70)
print("1. Базовые объёмы")
print(f"train rows {len(tr)}, unique items {tr.item_id.nunique()}, "
      f"unique query-texts {tr.search_query.nunique()}")
tr["qkey"] = (tr.search_query.astype(str) + "|" + tr.search_location_id.astype(str)
              + "|" + tr.search_infm_params_text.astype(str) + "|"
              + tr.search_is_delivery_search.astype(str) + "|"
              + tr.search_category.astype(str))
print("unique sessions (полный ключ запроса):", tr.qkey.nunique())
pos_per = tr.groupby("qkey").size()
print("позитивов на сессию:", pos_per.describe().round(2).to_dict())

# пересечение текстов запросов benchmark и train
tq = set(tr.search_query.astype(str)); bqt = set(bq.search_query.astype(str))
print(f"\nbenchmark query texts: {len(bqt)}, в train: {len(bqt & tq)} "
      f"({len(bqt & tq)/len(bqt):.1%})")
bqk = set(bq.search_query.astype(str) + "|" + bq.search_location_id.astype(str))
trk = set(tr.search_query.astype(str) + "|" + tr.search_location_id.astype(str))
print("query|loc ключ benchmark, совпадающих с train:",
      f"{len(bqk & trk)} ({len(bqk & trk)/len(bqk):.1%})")

print("=" * 70)
print("2. Категории")
print("search_category:", tr.search_category.value_counts().head(5).to_dict())
print("item_category_id:", tr.item_category_id.value_counts().head(5).to_dict())
print("cat_match:", (tr.search_category == tr.item_category_id).mean())
print("уникальных microcat в train:", tr.item_microcat_id.nunique(),
      "| в корпусе:", it.item_microcat_id.nunique())
print("search_infm_params непустой:", (tr.search_infm_params_text.str.len() > 0).mean(),
      "| примеры:", tr.search_infm_params_text[tr.search_infm_params_text.str.len() > 0].head(3).tolist())


print("=" * 70)
print("3. Локация: точное совпадение и иерархия")
lm = (tr.search_location_id == tr.item_location_id)
print(f"loc_match (train): {lm.mean():.4f}")
sl, il = tr.search_location_id.to_numpy(), tr.item_location_id.to_numpy()

mm = tr[~lm]
print("пар с несовпадением локации:", len(mm))
top = mm.groupby(["search_location_id", "item_location_id"]).size().sort_values(ascending=False).head(15)
print(top.to_string())

# асимметрия: является ли связь иерархической (S - родитель I)?
# если S родитель I, то запросы в локации I почти никогда не выбирают items из S
rev = tr.groupby(["item_location_id", "search_location_id"]).size()
pairs = mm.groupby(["search_location_id", "item_location_id"]).size().reset_index(name="c")
pairs = pairs.sort_values("c", ascending=False)
pairs["rev"] = [int(rev.get((b, a), 0))
                for a, b in zip(pairs.search_location_id, pairs.item_location_id)]
print("\nтоп направленных пар S->I с обратной частотой (признак иерархии):")
print(pairs.head(15).to_string())
print(f"доля направленных пар с обратными переходами: {(pairs.rev > 0).mean():.3f}")


print("=" * 70)
print("4. Геометрия: item_location_id -> координаты")
lat = pd.to_numeric(tr.item_latitude, errors="coerce")
lon = pd.to_numeric(tr.item_longitude, errors="coerce")
g = tr.assign(lat=lat, lon=lon).groupby("item_location_id").agg(
    lat=("lat", "mean"), lon=("lon", "mean"),
    lat_std=("lat", "std"), lon_std=("lon", "std"), n=("lat", "size"))
print("локаций в train:", len(g), "; медианный std lat:", g.lat_std.median(),
      "lon:", g.lon_std.median(), "; медианный размер:", g.n.median())
print("локаций с >100 items:", (g.n > 100).sum())

itl = it.assign(lat=pd.to_numeric(it.item_latitude, errors="coerce"),
                lon=pd.to_numeric(it.item_longitude, errors="coerce"))
cent = itl.groupby("item_location_id").agg(clat=("lat", "mean"), clon=("lon", "mean"))
print("локаций в корпусе:", len(cent))


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


# расстояние item -> центроид локации поиска (центроид позитивов train)
pos_cent = tr.assign(lat=lat, lon=lon).groupby("search_location_id").agg(
    slat=("lat", "mean"), slon=("lon", "mean"))
n = min(200000, len(tr))
d = [np.nan] * n
pc_idx = pos_cent.index
for j in range(n):
    s, i = sl[j], il[j]
    if s in pc_idx and i in cent.index:
        a, b = pos_cent.loc[s], cent.loc[i]
        d[j] = hav(a.slat, a.slon, b.clat, b.clon)
d = np.array(d, dtype=float)
fin = ~np.isnan(d)
print("\nрасстояние item -> центроид локации поиска (позитивы train), км:")
print(pd.Series(d[fin]).describe().round(1).to_dict())
print("доля <= 15 км:", (d[fin] <= 15).mean(), "| <= 30:", (d[fin] <= 30).mean(),
      "| <= 100:", (d[fin] <= 100).mean())


print("=" * 70)
print("5. Метаданные позитивов vs корпуса")


def summarize(name, s_pos, s_corp):
    print(f"{name}: позитивы mean={s_pos.mean():.3f} med={s_pos.median():.3f} | "
          f"корпус mean={s_corp.mean():.3f} med={s_corp.median():.3f}")


for col in ["item_rating", "item_rating_reviews_count"]:
    summarize(col, tr[col].astype(float), pd.to_numeric(it[col], errors="coerce").astype(float))
pr_pos = pd.to_numeric(tr.item_price, errors="coerce")
pr_corp = pd.to_numeric(it.item_price, errors="coerce")
summarize("log price", np.log1p(pr_pos.fillna(0)), np.log1p(pr_corp.fillna(0)))
summarize("phone_hidden", tr.item_is_phone_hidden.astype(float),
          it.item_is_phone_hidden.astype(float))
summarize("msg_forbidden", tr.item_is_message_forbidden.astype(float),
          it.item_is_message_forbidden.astype(float))

print("=" * 70)
print("6. search_infm_params_text vs item_infm_params_text")
tok = lambda s: set(re.findall(r"[а-яa-z0-9]+", str(s).lower()))
sp = tr.search_infm_params_text.map(tok)
ip = tr.item_infm_params_text.map(tok)
ov = np.fromiter((len(a & b) for a, b in zip(sp, ip)), dtype=np.int32, count=len(tr))
qlen = sp.map(len).to_numpy()
print("mean пересечение параметров:", ov.mean(), "| при пустых search-параметрах:",
      ov[qlen == 0].mean(), "| при непустых:", ov[qlen > 0].mean())
print("доля пар с ov>0 при непустых:", (ov[qlen > 0] > 0).mean())
fields = Counter()
for s in tr.search_infm_params_text.head(50000):
    for m in re.findall(r"(Вид услуги|Тип услуги|Место оказания услуг|Онлайн-запись|Рейтинг|Доставка)", str(s)):
        fields[m] += 1
print("поля в search_infm_params_text:", fields.most_common())

print("=" * 70)
print("7. Дубликаты item в train (популярность объявлений)")
pop = tr.item_id.value_counts()
print("items выбраны >1 раз:", (pop > 1).sum(), "из", len(pop),
      f"| max {pop.max()} | топ-5:", pop.head().to_dict())
print("покрытие корпуса позитивами:", tr.item_id.isin(set(it.item_id)).mean())

print("=" * 70)
print("8. Локации: сколько item_loc у одного search_loc (позитивы)")
s2i = tr.groupby("search_location_id").item_location_id.nunique()
print(s2i.describe().round(1).to_dict())
i2s = tr.groupby("item_location_id").search_location_id.nunique()
print("search_loc на один item_loc:", i2s.describe().round(1).to_dict())

print("=" * 70)
print("9. Разница распределений: benchmark-запросы vs train-сессии")
print("search_loc в benchmark есть в train:",
      round(bq.search_location_id.isin(set(tr.search_location_id)).mean(), 3))
print("query текст benchmark есть в train:",
      round(bq.search_query.isin(tq).mean(), 3))
bk_full = set(bq.search_query.astype(str) + "|" + bq.search_location_id.astype(str)
              + "|" + bq.search_infm_params_text.astype(str) + "|"
              + bq.search_is_delivery_search.astype(str) + "|"
              + bq.search_category.astype(str))
print("полный qkey benchmark есть в train:", round(len(bk_full & set(tr.qkey)) / len(bq), 3))
print("delivery в benchmark:", round(bq.search_is_delivery_search.mean(), 3),
      "| в train:", round(tr.search_is_delivery_search.mean(), 3))
