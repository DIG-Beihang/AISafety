# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2021-09-13
"""

from typing import NoReturn, Set

from .base import PreTransformationConstraint
from ..attacked_text import AttackedText


__all__ = [
    "MinWordLen",
]


class MinWordLen(PreTransformationConstraint):
    """A constraint that prevents modifications to words less than a certain word character-length."""

    def __init__(self, min_length: int) -> NoReturn:
        """
        :param min_length: Minimum word character-length needed for changes to be made to a word.
        """
        self.min_length = min_length

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        idxs = []
        for i, word in enumerate(current_text.words):
            if len(word) >= self.min_length:
                idxs.append(i)
        return set(idxs)
