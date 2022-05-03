# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-22
@LastEditTime: 2022-03-05
"""

from typing import Iterable, Set, NoReturn

import nltk

from .base import PreTransformationConstraint
from ..transformations import (
    Transformation,
    transformation_consists_of_word_substitutes,
)
from ..attacked_text import AttackedText


__all__ = [
    "StopwordModification",
]


class StopwordModification(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords."""

    def __init__(
        self, stopwords: Iterable[str] = None, language: str = "english"
    ) -> NoReturn:
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words(language))

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_stopword_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.stopwords:
                non_stopword_indices.add(i)
        return non_stopword_indices

    def check_compatibility(self, transformation: Transformation) -> bool:
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_substitutes(transformation)
