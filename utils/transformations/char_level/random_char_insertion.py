# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2022-04-17

"""

from typing import NoReturn, Optional, List, Any

from ..base import CharSubstitute
from ...misc import DEFAULTS


__all__ = [
    "RandomCharacterInsertion",
]


class RandomCharacterInsertion(CharSubstitute):
    """Transforms an input by inserting a random character."""

    __name__ = "RandomCharacterInsertion"

    def __init__(
        self,
        random_one: bool = True,
        skip_first_char: bool = False,
        skip_last_char: bool = False,
        **kwargs: Any
    ) -> NoReturn:
        """
        Args:
            random_one:
                Whether to return a single word with two characters swapped.
                If not, returns all possible options.
            skip_first_char:
                Whether to disregard perturbing the first character.
            skip_last_char:
                Whether to disregard perturbing the last character.
        """
        super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """Returns returns a list containing all possible words with 1 random character inserted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 1) if self.skip_last_char else len(word)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = DEFAULTS.RNG.integers(start_idx, end_idx)
            candidate_word = word[:i] + self._get_random_letter() + word[i:]
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx):
                candidate_word = word[:i] + self._get_random_letter() + word[i:]
                candidate_words.append(candidate_word)

        if num:
            candidate_words = candidate_words[:num]

        return candidate_words

    @property
    def deterministic(self) -> bool:
        return not self.random_one

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "random_one",
            "skip_first_char",
            "skip_last_char",
        ]
