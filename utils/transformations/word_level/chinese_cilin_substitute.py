# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-17
@LastEditTime: 2021-11-10
"""

import random
from typing import List, NoReturn

from ..base import WordSubstitute
from ... import assets


__all__ = [
    "ChineseCiLinSubstitute",
]


class ChineseCiLinSubstitute(WordSubstitute):
    """ """

    __name__ = "ChineseCiLinSubstitute"

    def __init__(self) -> NoReturn:
        """ """
        super().__init__()
        self.cilin_dict = assets.fetch("cilin")

    @property
    def deterministic(self) -> bool:
        return False

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """ """
        if word not in self.cilin_dict:
            return []
        sym_words = self.cilin_dict[word]
        ret = []
        for sym_word in sym_words:
            ret.append(sym_word)
        random.shuffle(ret)
        if num is not None:
            ret = ret[:num]
        return ret
