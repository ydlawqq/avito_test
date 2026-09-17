"""Текстовая предобработка: нормализация, токенизация, лемматизация (pymorphy3).

Пайплайн для BM25:
    сырой текст -> lower -> токены (буквы/цифры) -> леммы -> фильтр стоп-слов

Лемматизация дорогая (чистый Python), поэтому:
  * леммы кэшируются в словарь на процесс (словарь корпуса сильно меньше корпуса);
  * массовая обработка идёт через multiprocessing.Pool.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pymorphy3

# ---------------------------------------------------------------------------
# Стоп-слова
# ---------------------------------------------------------------------------

# Встроенный список — fallback, если nltk-корпус недоступен (нет интернета).
_FALLBACK_RU_STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же
    вы за бы по только ее мне было вот от меня еще нет о из ему теперь
    когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
    опять уж вам сказал ведь там потом себя ничего ей может они тут где
    есть надо ней для мы тебя их чем была сам чтоб без будто человек чего
    раз тоже себе под жизнь будет ж тогда кто этот говорил того потому
    этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется
    сейчас были куда зачем сказать всех никогда сегодня можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них
    какая много разве эту моя впрочем хорошо свою этой перед иногда
    лучше чуть том нельзя такой им более всегда конечно всю между это
    которые
    """.split()
)


def get_stopwords() -> frozenset[str]:
    """Стоп-слова для русского языка: nltk + встроенный fallback."""
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("russian")) | _FALLBACK_RU_STOPWORDS
    except LookupError:
        return _FALLBACK_RU_STOPWORDS


# ---------------------------------------------------------------------------
# Токенизация и лемматизация
# ---------------------------------------------------------------------------

# Токен: непрерывная последовательность русских/латинских букв или цифр.
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

_morph: pymorphy3.MorphAnalyzer | None = None
_lemma_cache: dict[str, str] = {}


def _get_morph() -> pymorphy3.MorphAnalyzer:
    """Ленивая инициализация анализатора (по одному на процесс)."""
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def tokenize(text: str) -> list[str]:
    """lowercase -> список токенов."""
    return TOKEN_RE.findall(text.lower())


def lemmatize_word(word: str) -> str:
    """Лемма одного слова с кэшированием (словарь на процесс)."""
    lemma = _lemma_cache.get(word)
    if lemma is None:
        parses = _get_morph().parse(word)
        lemma = parses[0].normal_form if parses else word
        _lemma_cache[word] = lemma
    return lemma


def lemmatize_text(text: str, stopwords: frozenset[str]) -> list[str]:
    """Полный пайплайн для одного текста: текст -> список лемм без стоп-слов."""
    return [lemmatize_word(t) for t in tokenize(text) if t not in stopwords]


def _lemmatize_batch(texts: list[str], stopwords: Sequence[str]) -> list[list[str]]:
    """Обработка пачки текстов внутри воркера multiprocessing.Pool."""
    sw = frozenset(stopwords)
    return [lemmatize_text(t, sw) for t in texts]


def lemmatize_many(
    texts: list[str],
    stopwords: frozenset[str],
    n_jobs: int = 1,
    batch_size: int = 500,
    progress: bool = True,
) -> list[list[str]]:
    """Лемматизация большого списка текстов (multiprocessing при n_jobs > 1)."""
    batches = [
        (texts[i : i + batch_size], tuple(stopwords))
        for i in range(0, len(texts), batch_size)
    ]
    results: list[list[list[str]]] = []

    if n_jobs > 1:
        import multiprocessing as mp

        with mp.Pool(n_jobs) as pool:
            it = pool.starmap(_lemmatize_batch, batches, chunksize=1)
            results = list(it)
    else:
        for batch, sw in batches:
            results.append(_lemmatize_batch(batch, sw))

    out: list[list[str]] = []
    for chunk in results:
        out.extend(chunk)

    from tqdm import tqdm

    _ = tqdm  # tqdm используется вызывающими скриптами
    return out


def join_tokens(tokens: Iterable[str]) -> str:
    """Склеить токены обратно в строку (удобно для хранения в parquet)."""
    return " ".join(tokens)
