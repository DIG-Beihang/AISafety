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


__all__ = [
    "WordContractSubstitute",
]


_EXTENSION_MAP = fetch("checklist", keys="EXTENSION_MAP")


class WordContractSubstitute(WordSubstitute):
    """Transforms an input by performing extension on recognized combinations."""

    __name__ = "WordContractSubstitute"

    reverse_contraction_map = {v: k for k, v in _EXTENSION_MAP.items()}

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
        indices_to_modify = sorted(indices_to_modify)

        # search for every 2-words combination in reverse_contraction_map
        for idx, word_idx in enumerate(indices_to_modify[:-1]):
            next_idx = indices_to_modify[idx + 1]
            if (idx + 1) != next_idx:
                continue
            word = words[word_idx]
            next_word = words[next_idx]

            # generating the words to search for
            key = " ".join([word, next_word])

            # when a possible contraction is found in map, contract the current text
            if key in self.reverse_contraction_map:
                transformed_text = current_text.replace_word_at_index(
                    idx, self.reverse_contraction_map[key]
                )
                transformed_text = transformed_text.delete_word_at_index(next_idx)
                transformed_texts.append(transformed_text)

        return transformed_texts

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """ """
        raise ValueError(f"{self.__name__} 中请勿调用 _get_candidates 方法")
