# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-11-10
"""

from typing import List

from nltk.corpus import wordnet as wn

from ..base import WordSubstitute
from ...strings import normalize_pos_tag, UNIVERSAL_POSTAG
from ...strings_en import tokenize


__all__ = [
    "WordNetSubstitute",
]


class WordNetSubstitute(WordSubstitute):
    """WordNet synonym substitute"""

    _VALID_POS = {
        UNIVERSAL_POSTAG.NOUN: "n",
        UNIVERSAL_POSTAG.VERB: "v",
        UNIVERSAL_POSTAG.ADJ: "a",
        UNIVERSAL_POSTAG.ADV: "r",
    }
    __name__ = "WordNetSubstitute"

    def __init__(self):
        super().__init__()

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """ """
        try:
            pos_tag = normalize_pos_tag(pos_tag)
        except Exception:
            pos_tag = None
        if pos_tag is None or pos_tag not in self._VALID_POS:
            return [word]

        pos = self._VALID_POS[pos_tag]
        synonyms = []
        for synset in wn.synsets(word, pos=pos, lang="eng"):
            for lemma in synset.lemma_names("eng"):
                if lemma == word or "_" in lemma or len(tokenize(lemma)) != 1:
                    continue
                synonyms.append(lemma)
        if num:
            return synonyms[:num]
        return synonyms
