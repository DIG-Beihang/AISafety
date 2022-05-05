# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-12
@LastEditTime: 2021-09-13

基于Language Model的限制条件的抽象基类

主要基于TextAttack的LanguageModelConstraint实现
"""

from abc import ABC, abstractmethod
from typing import List, Any, NoReturn, Optional, Sequence

from ..base import Constraint
from ...attacked_text import AttackedText


__all__ = [
    "LanguageModelBase",
]


class LanguageModelBase(Constraint, ABC):
    """ """

    __name__ = "LanguageModelBase"

    """Determines if two sentences have a swapped word that has a similar
    probability according to a language model."""

    def __init__(
        self,
        max_log_prob_diff: Optional[float] = None,
        compare_against_original: bool = True,
    ) -> NoReturn:
        """
        Args:
            max_log_prob_diff:
                the maximum decrease in log-probability in swapped words from `x` to `x_adv`
            compare_against_original:
                If `True`, compare new `x_adv` against the original `x`.
                Otherwise, compare it against the previous `x_adv`.
        """
        if max_log_prob_diff is None:
            raise ValueError("Must set max_log_prob_diff")
        self.max_log_prob_diff = max_log_prob_diff
        super().__init__(compare_against_original)

    @abstractmethod
    def get_log_probs_at_index(
        self, text_list: Sequence[AttackedText], word_index: int
    ) -> Any:
        """Gets the log-probability of items in `text_list` at index
        `word_index` according to a language model."""
        raise NotImplementedError

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply language model constraint without `newly_modified_indices`"
            )

        for i in indices:
            probs = self.get_log_probs_at_index((reference_text, transformed_text), i)
            if len(probs) != 2:
                raise ValueError(
                    f"Error: get_log_probs_at_index returned {len(probs)} values for 2 inputs"
                )
            ref_prob, transformed_prob = probs
            if transformed_prob <= ref_prob - self.max_log_prob_diff:
                return False

        return True

    def extra_repr_keys(self) -> List[str]:
        return ["max_log_prob_diff"] + super().extra_repr_keys()
