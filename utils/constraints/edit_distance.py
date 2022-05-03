# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-27
@LastEditTime: 2021-08-18
"""

import Levenshtein

from .base import Constraint
from ..attacked_text import AttackedText


__all__ = [
    "EditDistance",
]


class EditDistance(Constraint):
    """
    编辑距离限制 (Levenshtein distance)
    """

    def __init__(self, max_edit_distance: int, compare_against_original: bool = True):
        super().__init__(compare_against_original)
        if not isinstance(max_edit_distance, int):
            raise TypeError("max_edit_distance must be an int")
        self.max_edit_distance = max_edit_distance

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        edit_distance = Levenshtein.distance(reference_text.text, transformed_text.text)
        return edit_distance <= self.max_edit_distance

    def extra_repr_keys(self):
        return ["max_edit_distance"] + super().extra_repr_keys()
