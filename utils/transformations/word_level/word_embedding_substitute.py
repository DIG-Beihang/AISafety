# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-20
@LastEditTime: 2021-09-08
"""

from typing import Any, List, NoReturn, Optional

from ..base import WordSubstitute
from ...word_embeddings import WordEmbedding


__all__ = [
    "WordEmbeddingSubstitute",
]


class WordEmbeddingSubstitute(WordSubstitute):
    """Transforms an input by replacing its words with synonyms in the word embedding space.

    Based on paper: `<arxiv.org/abs/1603.00892>`_

    Paper title: Counter-fitting Word Vectors to Linguistic Constraints
    """

    __name__ = "WordEmbeddingSubstitute"

    def __init__(
        self, embedding: WordEmbedding, max_candidates: int = 15, **kwargs: Any
    ) -> NoReturn:
        """
        @param {
            embedding: Wrapper for word embedding
            max_candidates: maximum number of synonyms to pick
        }
        @return: None
        """
        # super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.embedding = embedding
        self.verbose = kwargs.get("verbose", 0)

    def _get_candidates(
        self, word: str, pos_tag: Optional[str] = None, num: Optional[int] = None
    ) -> List[str]:
        """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.embedding.word2index(word.lower())
            nnids = self.embedding.nearest_neighbours(word_id, self.max_candidates)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.embedding.index2word(nbr_id)
                candidate_words.append(_recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []
        if self.verbose >= 2:
            print(
                f"using {self.__name__}, one get candidate_words\n{candidate_words}\ntransformed from `{word}`"
            )

    def extra_repr_keys(self) -> List[str]:
        return [
            "max_candidates",
            "embedding",
        ]


def _recover_word_case(word: str, reference_word: str) -> str:
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
