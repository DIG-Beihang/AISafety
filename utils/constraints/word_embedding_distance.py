# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-19
@LastEditTime: 2021-09-06
"""

from typing import Union, NoReturn, List

from .base import Constraint
from ..word_embeddings import AbstractWordEmbedding, WordEmbedding
from ..attacked_text import AttackedText
from ..transformations.base import (
    Transformation,
    transformation_consists_of_word_substitutes,
)


__all__ = [
    "WordEmbeddingDistance",
]


class WordEmbeddingDistance(Constraint):
    """词嵌入向量的余弦距离限制"""

    def __init__(
        self,
        embedding: WordEmbedding,
        include_unknown_words: bool = True,
        min_cos_sim: float = None,
        max_mse_dist: float = None,
        cased: bool = False,
        compare_against_original: bool = True,
    ) -> NoReturn:
        """
        @description: Grammar error number constraint
        @param {
            embedding: Wrapper for word embedding.
            include_unknown_words: Whether or not the constraint is fulfilled if the embedding of x or x_adv is unknown.
            min_cos_sim: The minimum cosine similarity between word embeddings.
            max_mse_dist: The maximum euclidean distance between word embeddings.
            cased: Whether embedding supports uppercase & lowercase (defaults to False, or just lowercase).
            compare_against_original:
                If `True`, compare new `x_adv` against the original `x`. Otherwise, compare it against the previous `x_adv`.
        }
        @return: None
        """
        super().__init__(compare_against_original)
        self.include_unknown_words = include_unknown_words
        self.cased = cased

        if bool(min_cos_sim) == bool(max_mse_dist):
            raise ValueError("You must choose either `min_cos_sim` or `max_mse_dist`.")
        self.min_cos_sim = min_cos_sim
        self.max_mse_dist = max_mse_dist

        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `AbstractWordEmbedding`."
            )
        self.embedding = embedding

    def get_cos_sim(self, a: Union[str, int], b: Union[str, int]) -> float:
        """Returns the cosine similarity of words with IDs a and b."""
        return self.embedding.get_cos_sim(a, b)

    def get_mse_dist(self, a: Union[str, int], b: Union[str, int]) -> float:
        """Returns the MSE distance of words with IDs a and b."""
        return self.embedding.get_mse_dist(a, b)

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """Returns true if (``transformed_text`` and ``reference_text``) are
        closer than ``self.min_cos_sim`` or ``self.max_mse_dist``."""
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        # FIXME The index i is sometimes larger than the number of tokens - 1
        if any(
            i >= len(reference_text.words) or i >= len(transformed_text.words)
            for i in indices
        ):
            return False

        for i in indices:
            ref_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]

            if not self.cased:
                # If embedding vocabulary is all lowercase, lowercase words.
                ref_word = ref_word.lower()
                transformed_word = transformed_word.lower()

            try:
                ref_id = self.embedding.word2index(ref_word)
                transformed_id = self.embedding.word2index(transformed_word)
            except KeyError:
                # This error is thrown if x or x_adv has no corresponding ID.
                if self.include_unknown_words:
                    continue
                return False

            # Check cosine distance.
            if self.min_cos_sim:
                cos_sim = self.get_cos_sim(ref_id, transformed_id)
                if cos_sim < self.min_cos_sim:
                    return False
            # Check MSE distance.
            if self.max_mse_dist:
                mse_dist = self.get_mse_dist(ref_id, transformed_id)
                if mse_dist > self.max_mse_dist:
                    return False

        return True

    def check_compatibility(self, transformation: Transformation) -> bool:
        """WordEmbeddingDistance requires a word being both deleted and
        inserted at the same index in order to compare their embeddings,
        therefore it's restricted to word swaps."""
        return transformation_consists_of_word_substitutes(transformation)

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        if self.min_cos_sim is None:
            metric = "max_mse_dist"
        else:
            metric = "min_cos_sim"
        return [
            "embedding",
            metric,
            "cased",
            "include_unknown_words",
        ] + super().extra_repr_keys()
