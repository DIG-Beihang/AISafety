# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2022-04-17

"""

from typing import NoReturn, List, Tuple, Any, Sequence, Optional

from ..base import WordSubstitute
from ...assets import fetch
from ...attacked_text import AttackedText
from ...misc import consecutive_groups, DEFAULTS
from ...strings import LANGUAGE


__all__ = [
    "WordChangeLocSubstitute",
]


_NAMED_ENTITIES = fetch("checklist", keys="NAMED_ENTITIES")


class WordChangeLocSubstitute(WordSubstitute):
    """ """

    __name__ = "WordChangeLocSubstitute"

    def __init__(
        self, n: int = 3, confidence_score: float = 0.7, **kwargs: Any
    ) -> NoReturn:
        """Transformation that changes recognized locations of a sentence to
        another location that is given in the location map.

        :param n: Number of new locations to generate
        :param confidence_score: Location will only be changed if it's above the confidence score
        """
        super().__init__(**kwargs)
        self.n = n
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
        ), f"{self.__name__} currently only supports English text"
        words = current_text.words
        location_idx = []

        for i in indices_to_modify:
            tag = current_text.ner_of_word_index(i)
            if "LOC" in tag.value and tag.score > self.confidence_score:
                location_idx.append(i)

        # Combine location idx and words to a list ([0] is idx, [1] is location name)
        # For example, [1,2] to [ [1,2] , ["New York"] ]
        location_idx = [list(group) for group in consecutive_groups(location_idx)]
        location_words = idx_to_words(location_idx, words)

        transformed_texts = []
        for location in location_words:
            idx = location[0]
            word = location[1].capitalize()
            replacement_words = self._get_candidates(word)
            for r in replacement_words:
                if r == word:
                    continue
                text = current_text

                # if original location is more than a single word, remain only the starting word
                if len(idx) > 1:
                    index = idx[1]
                    for i in idx[1:]:
                        text = text.delete_word_at_index(index)

                # replace the starting word with new location
                text = text.replace_word_at_index(idx[0], r)

                transformed_texts.append(text)
        return transformed_texts

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """Return a list of new locations, with the choice of country,
        nationality, and city."""
        if word in _NAMED_ENTITIES["country"]:
            return DEFAULTS.RNG.choice(_NAMED_ENTITIES["country"], self.n).tolist()
        elif word in _NAMED_ENTITIES["nationality"]:
            return DEFAULTS.RNG.choice(_NAMED_ENTITIES["nationality"], self.n).tolist()
        elif word in _NAMED_ENTITIES["city"]:
            return DEFAULTS.RNG.choice(_NAMED_ENTITIES["city"], self.n).tolist()
        return []


def idx_to_words(ls: List[List[int]], words: Sequence[str]) -> List[Tuple[list, str]]:
    """Given a list generated from cluster_idx, return a list that contains
    sub-list (the first element being the idx, and the second element being the
    words corresponding to the idx)"""

    output = []
    for sub_ls in ls:
        word = words[sub_ls[0]]
        for idx in sub_ls[1:]:
            word = " ".join([word, words[idx]])
        output.append([sub_ls, word])
    return output
