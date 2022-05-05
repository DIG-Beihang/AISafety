# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-20
@LastEditTime: 2022-04-17

"""

from typing import NoReturn, Optional, List, Any

from ..base import CharSubstitute
from ...misc import DEFAULTS


__all__ = [
    "RandomCharacterSubstitute",
]


class RandomCharacterSubstitute(CharSubstitute):
    """Transforms an input by deleting its characters."""

    __name__ = "RandomCharacterSubstitute"

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
        """Returns returns a list containing all possible words with 1 letter deleted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        if self.random_one:
            i = DEFAULTS.RNG.integers(0, len(word))
            candidate_word = word[:i] + self._get_random_letter() + word[i + 1 :]
            candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                candidate_word = word[:i] + self._get_random_letter() + word[i + 1 :]
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
