# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2021-09-13
"""

from typing import List, Sequence, Optional

from ..base import WordSubstitute
from ...assets import fetch
from ...attacked_text import AttackedText
from ...strings import LANGUAGE


__all__ = ["WordExtendSubstitute"]


_EXTENSION_MAP = fetch("checklist", keys="EXTENSION_MAP")


class WordExtendSubstitute(WordSubstitute):
    """Transforms an input by performing extension on recognized combinations."""

    __name__ = "WordExtendSubstitute"

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """Return all possible transformed sentences, each with one extension."""
        assert (
            current_text.language == LANGUAGE.ENGLISH
        ), f"{self.__name__} currently only supports English text"
        transformed_texts = []
        words = current_text.words
        for idx in indices_to_modify:
            word = words[idx]
            # expend when word in map
            if word in _EXTENSION_MAP:
                expanded = _EXTENSION_MAP[word]
                transformed_text = current_text.replace_word_at_index(idx, expanded)
                transformed_texts.append(transformed_text)

        return transformed_texts

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """ """
        raise ValueError(f"{self.__name__} 中请勿调用 _get_candidates 方法")
