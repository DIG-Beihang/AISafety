# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-22
@LastEditTime: 2022-03-19

注意：
ChineseWordNetSubstitute可能导致
__eq__方法未加depth参数的AttackText的比较陷入死循环
"""

import random
from typing import List, Optional

from nltk.corpus import wordnet as wn

from ..base import WordSubstitute
from ...strings import normalize_pos_tag, UNIVERSAL_POSTAG


__all__ = [
    "ChineseWordNetSubstitute",
]


class ChineseWordNetSubstitute(WordSubstitute):
    """ChineseWordNet synonym substitute"""

    _VALID_POS = {
        UNIVERSAL_POSTAG.NOUN: "n",
        UNIVERSAL_POSTAG.VERB: "v",
        UNIVERSAL_POSTAG.ADJ: "a",
        UNIVERSAL_POSTAG.ADV: "r",
    }
    __name__ = "ChineseWordNetSubstitute"

    def __init__(self):
        super().__init__()

    @property
    def deterministic(self) -> bool:
        return False

    def _get_candidates(
        self, word: str, pos_tag: Optional[str] = None, num: Optional[int] = None
    ) -> List[str]:
        """ """
        pos_tag = normalize_pos_tag(pos_tag)
        if pos_tag is None or pos_tag not in self._VALID_POS:
            return [word]

        pos = self._VALID_POS[pos_tag]
        synonyms = []
        for synset in wn.synsets(word, pos=pos, lang="cmn"):
            for lemma in synset.lemma_names("cmn"):
                if lemma == word:
                    continue
                synonyms.append(lemma)
        random.shuffle(synonyms)
        if num:
            return synonyms[:num]
        return synonyms
