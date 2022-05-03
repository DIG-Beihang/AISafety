# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-13
"""

import functools
from typing import Any, NoReturn, Sequence, List

import torch

from .sentence_encoder_base import SentenceEncoderBase
from ...word_embeddings import WordEmbedding
from ...strings import normalize_language, LANGUAGE
from ...strings_cn import words_from_text_cn
from ...strings_en import tokenize


class ThoughtVector(SentenceEncoderBase):
    """A constraint on the distance between two sentences' thought vectors.

    Args:
        word_embedding: The word embedding to use
    """

    def __init__(
        self, language: str, embedding: WordEmbedding, **kwargs: Any
    ) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self._tokenizer = {
            LANGUAGE.CHINESE: words_from_text_cn,
            LANGUAGE.ENGLISH: tokenize,
        }[self._language]
        self.word_embedding = embedding
        super().__init__(**kwargs)

    def clear_cache(self):
        self._get_thought_vector.cache_clear()

    @functools.lru_cache(maxsize=2**10)
    def _get_thought_vector(self, text: str) -> torch.Tensor:
        """Sums the embeddings of all the words in ``text`` into a "thought vector"."""
        embeddings = []
        for word in self._tokenizer(text):
            embedding = self.word_embedding[word]
            if embedding is not None:  # out-of-vocab words do not have embeddings
                embeddings.append(embedding)
        embeddings = torch.tensor(embeddings)
        return torch.mean(embeddings, dim=0)

    def encode(self, raw_text_list: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._get_thought_vector(text) for text in raw_text_list])

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys."""
        return ["word_embedding"] + super().extra_repr_keys()
