# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2022-04-17

"""

import re
from typing import NoReturn, List, Any, Sequence, Optional

from ..base import WordSubstitute
from ...assets import fetch
from ...attacked_text import AttackedText
from ...strings import NERTAG, LANGUAGE
from ...misc import DEFAULTS


__all__ = [
    "WordChangeNameSubstitute",
]


_PERSON_NAMES = fetch("checklist", keys="PERSON_NAMES")


class WordChangeNameSubstitute(WordSubstitute):
    """ """

    __name__ = "WordChangeNameSubstitute"

    def __init__(
        self,
        num_name_replacements: int = 3,
        first_only: bool = False,
        last_only: bool = False,
        confidence_score: float = 0.7,
        **kwargs: Any,
    ) -> NoReturn:
        """Transforms an input by replacing names of recognized name entity.

        :param n: Number of new names to generate per name detected
        :param first_only: Whether to change first name only
        :param last_only: Whether to change last name only
        :param confidence_score: Name will only be changed when it's above confidence score
        """
        super().__init__(**kwargs)
        self.num_name_replacements = num_name_replacements
        if first_only & last_only:
            raise ValueError("first_only and last_only cannot both be true")
        self.first_only = first_only
        self.last_only = last_only
        self.confidence_score = confidence_score

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        assert (
            current_text.language == LANGUAGE.ENGLISH
        ), f"{self.__name__} currently only works on English text"
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = current_text.words[i].capitalize()
            word_to_replace_ner = current_text.ner_of_word_index(i)
            replacement_words = self._get_candidates(
                word_to_replace, word_to_replace_ner
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))

        return transformed_texts

    def _get_candidates(
        self, word: str, ner_tag: NERTAG, num: Optional[int] = None
    ) -> List[str]:
        """ """
        replacement_words = []
        if (
            len(re.findall("PER", ner_tag.value)) > 0
            and ner_tag.score >= self.confidence_score
            and not self.last_only
        ):
            replacement_words = self._get_firstname(word)
        elif (
            len(re.findall("PER", ner_tag.value)) > 0
            and ner_tag.score >= self.confidence_score
            and not self.first_only
        ):
            replacement_words = self._get_lastname(word)
        return replacement_words

    def _get_lastname(self, word: str) -> List[str]:
        """Return a list of random last names."""
        return DEFAULTS.RNG.choice(
            _PERSON_NAMES["last"], self.num_name_replacements
        ).tolist()

    def _get_firstname(self, word: str) -> List[str]:
        """Return a list of random first names."""
        return DEFAULTS.RNG.choice(
            _PERSON_NAMES["first"], self.num_name_replacements
        ).tolist()
