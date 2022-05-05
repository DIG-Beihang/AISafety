# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2022-04-17

"""

from typing import NoReturn, List, Tuple, Any, Sequence, Optional, Union

from num2words import num2words
from word2number import w2n

from ..base import WordSubstitute
from ...attacked_text import AttackedText
from ...misc import consecutive_groups, DEFAULTS
from ...strings import LANGUAGE


__all__ = [
    "WordChangeNumSubstitute",
]


class WordChangeNumSubstitute(WordSubstitute):
    """ """

    __name__ = "WordChangeNumSubstitute"

    def __init__(self, max_change: int = 1, n: int = 3, **kwargs: Any) -> NoReturn:
        """A transformation that recognizes numbers in sentence, and returns
        sentences with altered numbers.

        :param max_change: Maximum percent of change (1 being 100%)
        :param n: Numbers of new numbers to generate
        """
        super().__init__(**kwargs)
        self.max_change = max_change
        self.n = n

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
        num_idx = []
        num_words = []

        # find indexes of alphabetical words
        for idx in indices_to_modify:
            word = words[idx].lower()
            for number in _STR_NUM[current_text.language]:
                if number in word:
                    if word in ["point", "and"]:
                        if 0 < idx and (idx - 1) in num_idx:
                            num_idx.append(idx)
                    else:
                        num_idx.append(idx)
                    break

            if word.isdigit():
                num_words.append([[idx], word])

        # cluster adjacent indexes to get whole number
        num_idx = [list(group) for group in consecutive_groups(num_idx)]
        num_words += idx_to_words(num_idx, words)

        # replace original numbers with new numbers
        transformed_texts = []
        for (idx, word) in num_words:
            replacement_words = self._get_candidates(word)
            for r in replacement_words:
                if r == word:
                    continue
                text = current_text.replace_word_at_index(idx[0], str(r))
                if len(idx) > 1:
                    index = idx[1]
                    for i in idx[1:]:
                        text = text.delete_word_at_index(index)
                transformed_texts.append(text)
        return transformed_texts

    def _get_candidates(self, word: str, pos_tag: str = None) -> List[Union[int, str]]:
        """Given a word, try altering the value if the word is a number return
        in digits if word is given in digit, return in alphabetical form if
        word is given in alphabetical form."""

        if word.isdigit():
            num = float(word)
            return self._alter_number(num)
        else:
            try:
                num = w2n.word_to_num(word)
                num_list = self._alter_number(num)
                return [num2words(n) for n in num_list]
            except ValueError:
                return []

    def _alter_number(self, num: float) -> List[int]:
        """helper function of _get_candidates, replace a number with another
        random number within the range of self.max_change."""
        if num not in [0, 2, 4]:
            change = int(num * self.max_change) + 1
            if num >= 0:
                num_list = DEFAULTS.RNG.integers(
                    max(num - change, 1), num + change, self.n
                ).tolist()
            else:
                num_list = DEFAULTS.RNG.integers(
                    num - change, min(0, num + change), self.n
                ).tolist()
            return num_list
        return []


def idx_to_words(ls: List[List[int]], words: Sequence[str]) -> List[Tuple[list, str]]:
    """Given a list generated from cluster_idx, return a list that contains
    sub-list (the first element being the idx, and the second element being the
    words corresponding to the idx)"""

    output = []
    for cluster in ls:
        word = words[cluster[0]]
        for idx in cluster[1:]:
            word = " ".join([word, words[idx]])
        output.append([cluster, word])
    return output


_STR_NUM = {
    LANGUAGE.ENGLISH: [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "point",
        "and",
    ],
    LANGUAGE.CHINESE: list("零一二三四五六七八九十百千万亿点两"),
}
