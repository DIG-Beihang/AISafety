# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-17
@LastEditTime: 2021-09-21
"""

from typing import NoReturn, List, Optional

from ..base import CharSubstitute
from ... import assets


class ChineseFYHCharSubstitute(CharSubstitute):
    """ """

    def __init__(self) -> NoReturn:
        super().__init__()
        self.tra_dict, self.var_dict, self.hot_dict = assets.fetch("fyh")

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        res = set()
        for char in word:
            if any(
                [char in self.tra_dict, char in self.var_dict, char in self.hot_dict]
            ):
                if char in self.tra_dict:
                    res = res.union(word.replace(char, self.tra_dict[char]))
                if char in self.var_dict:
                    res = res.union(word.replace(char, self.var_dict[char]))
                if char in self.hot_dict:
                    res = res.union(word.replace(char, self.hot_dict[char]))
            res = list(res)
        if num is not None:
            res = res[:num]
        return res
