# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-08
"""

import math
from typing import Optional, NoReturn, List

from .base import Constraint
from ...utils.attacked_text import AttackedText


class MaxWordsPerturbed(Constraint):
    """A constraint representing a maximum allowed perturbed words."""

    def __init__(
        self,
        max_num_words: Optional[int] = None,
        max_percent: Optional[float] = None,
        compare_against_original: bool = True,
    ) -> NoReturn:
        """
        Args:
            max_num_words:
                Maximum number of perturbed words allowed.
            max_percent:
                Maximum percentage of words allowed to be perturbed.
            compare_against_original:
                If `True`, compare new `x_adv` against the original `x`.
                Otherwise, compare it against the previous `x_adv`.
        """
        super().__init__(compare_against_original)
        if not compare_against_original:
            raise ValueError(
                "Cannot apply constraint MaxWordsPerturbed with `compare_against_original=False`"
            )

        if (max_num_words is None) and (max_percent is None):
            raise ValueError("must set either `max_percent` or `max_num_words`")
        if max_percent and not (0 <= max_percent <= 1):
            raise ValueError("max perc must be between 0 and 1")
        self.max_num_words = max_num_words
        self.max_percent = max_percent

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """ """
        num_words_diff = len(transformed_text.all_words_diff(reference_text))
        if self.max_percent:
            min_num_words = min(len(transformed_text.words), len(reference_text.words))
            max_words_perturbed = math.ceil(min_num_words * (self.max_percent))
            max_percent_met = num_words_diff <= max_words_perturbed
        else:
            max_percent_met = True
        if self.max_num_words:
            max_num_words_met = num_words_diff <= self.max_num_words
        else:
            max_num_words_met = True

        return max_percent_met and max_num_words_met

    def extra_repr_keys(self) -> List[str]:
        metric = []
        if self.max_percent is not None:
            metric.append("max_percent")
        if self.max_num_words is not None:
            metric.append("max_num_words")
        return metric + super().extra_repr_keys()
