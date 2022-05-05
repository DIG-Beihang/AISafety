# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-29

Glove Embedding，Word CNN与LSTM都需要用到的词嵌入层
"""

import os
from typing import NoReturn, Sequence

import numpy as np
import torch
from torch import nn as nn

from utils._download_data import download_if_needed
from utils.misc import nlp_cache_dir


__all__ = [
    "EmbeddingLayer",
    "GloveEmbeddingLayer",
]


class EmbeddingLayer(nn.Module):
    """A layer of a model that replaces word IDs with their embeddings.

    This is a useful abstraction for any nn.module which wants to take word IDs
    (a sequence of text) as input layer but actually manipulate words'
    embeddings.

    Requires some pre-trained embedding with associated word IDs.
    """

    __name__ = "EmbeddingLayer"

    def __init__(
        self,
        n_d: int = 100,
        embedding_matrix: np.ndarray = None,
        word_list: Sequence[str] = None,
        oov: str = "<oov>",
        pad: str = "<pad>",
        normalize: bool = True,
    ) -> NoReturn:
        super().__init__()
        word2id = {}
        if embedding_matrix is not None:
            for word in word_list:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            # logger.debug(f"{len(word2id)} pre-trained word embeddings loaded.\n")

            n_d = len(embedding_matrix[0])

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        weight = self.embedding.weight
        weight.data[: len(word_list)].copy_(torch.from_numpy(embedding_matrix))
        # logger.debug(f"EmbeddingLayer shape: {weight.size()}")

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)


class GloveEmbeddingLayer(EmbeddingLayer):
    """Pre-trained Global Vectors for Word Representation (GLOVE) vectors. Uses
    embeddings of dimension 200.

    GloVe is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed on aggregated global
    word-word co-occurrence statistics from a corpus, and the resulting
    representations showcase interesting linear substructures of the word
    vector space.


    GloVe: Global Vectors for Word Representation. (Jeffrey Pennington,
        Richard Socher, and Christopher D. Manning. 2014.)
    """

    __name__ = "GloveEmbeddingLayer"
    EMBEDDING_PATH = os.path.join(nlp_cache_dir, "glove200")

    def __init__(self, emb_layer_trainable: bool = True) -> NoReturn:
        """ """
        glove_path = download_if_needed(
            uri="glove200",
            source="aitesting",
            dst_dir=nlp_cache_dir,
        )
        glove_word_list_path = os.path.join(glove_path, "glove.wordlist.npy")
        word_list = np.load(glove_word_list_path)
        glove_matrix_path = os.path.join(glove_path, "glove.6B.200d.mat.npy")
        embedding_matrix = np.load(glove_matrix_path)
        super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
        self.embedding.weight.requires_grad = emb_layer_trainable
