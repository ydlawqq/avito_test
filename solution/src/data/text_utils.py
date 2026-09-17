"""Очистка и лемматизация русскоязычных текстов для BM25-пайплайна."""

from __future__ import annotations

import math
import re

from pymorphy3 import MorphAnalyzer

# ё -> е для устойчивости к разнобою в написании
_TRANS = str.maketrans({"ё": "е", "Ё": "Е"})
# Всё, что не буква/цифра/пробел (включая «умные» кавычки, тире, дефисы, _) -> пробел
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Приводит сырой текст к нижнему регистру, убирает пунктуацию и лишние пробелы.

    - None/NaN -> ""
    - "баня на дровах «Прованс»!" -> "баня на дровах прованс"
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text).lower().translate(_TRANS)
    s = _NON_WORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def lemmatize_text(norm_text: str, morph: MorphAnalyzer, min_len: int = 2) -> str:
    """Лемматизирует уже нормализованную строку (пробельный токенизатор).

    Args:
        norm_text: результат normalize_text.
        morph: экземпляр pymorphy3.MorphAnalyzer.
        min_len: токены короче отбрасываются (убирает однобуквенные союзы/предлоги).

    Returns:
        Строка из лемм, разделённых пробелом. Числа и незнакомые слова остаются как есть.
    """
    if not norm_text:
        return ""
    lemmas: list[str] = []
    for token in norm_text.split():
        if len(token) < min_len:
            continue
        try:
            lemmas.append(morph.parse(token)[0].normal_form)
        except (ValueError, IndexError):
            lemmas.append(token)
    return " ".join(lemmas)