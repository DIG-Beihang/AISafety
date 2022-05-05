# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-19
@LastEditTime: 2022-03-19

词嵌入模型以及相应的距离度量，
主要基于TextAttack的AttackedText类进行的实现
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import importlib
from typing import Union, List, Any, NoReturn, Optional

import numpy as np
import torch

from .misc import default_device, nlp_cache_dir
from .strings import ReprMixin
from ._download_data import download_if_needed


__all__ = [
    "WordEmbedding",
    "GensimWordEmbedding",
    "ChineseWord2Vec",
    "CounterFittedEmbedding",
]


class AbstractWordEmbedding(ReprMixin, ABC):
    """Abstract class representing word embedding.

    This class specifies all the methods that is required to be defined
    so that it can be used for transformation and constraints.

    For custom word embedding, please create a
    class that inherits this class and implement the required methods.
    However, please first check if you can use `WordEmbedding` class,
    which has a lot of internal methods implemented.
    """

    __name__ = "AbstractWordEmbedding"

    @abstractmethod
    def __getitem__(self, index: Union[str, int]) -> np.ndarray:
        """Gets the embedding vector for word/id
        Args:
            index: `index` can either be word or integer representing the id of the word.
        Returns:
            vector: 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mse_dist(self, a: Union[str, int], b: Union[str, int]) -> float:
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: MSE (L2) distance
        """
        raise NotImplementedError

    @abstractmethod
    def get_cos_sim(self, a: Union[str, int], b: Union[str, int]) -> float:
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: cosine similarity
        """
        raise NotImplementedError

    @abstractmethod
    def word2index(self, word: str) -> int:
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word
        Returns:
            index
        """
        raise NotImplementedError

    @abstractmethod
    def index2word(self, index: int) -> str:
        """
        Convert index to corresponding word
        Args:
            index
        Returns:
            word
        """
        raise NotImplementedError

    @abstractmethod
    def nearest_neighbours(self, index: int, topn: int) -> List[int]:
        """
        Get top-N nearest neighbours for a word
        Args:
            index: ID of the word for which we're finding the nearest neighbours
            topn: Used for specifying N nearest neighbours
        Returns:
            neighbours: List of indices of the nearest neighbours
        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        return []


class WordEmbedding(AbstractWordEmbedding):
    """Object for loading word embeddings and related distances"""

    __name__ = "WordEmbedding"

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        word2index: Any,
        index2word: Any,
        nn_matrix: np.ndarray = None,
        device: torch.device = None,
    ) -> NoReturn:
        """
        @param {
            emedding_matrix: 2-D array of shape N x D where N represents size of vocab and D is the dimension of embedding vectors.
            word2index: dictionary (or a similar object) that maps word to its index with in the embedding matrix.
            index2word: dictionary (or a similar object) that maps index to its word.
            nn_matrix: Matrix for precomputed nearest neighbours. It should be a 2-D integer array of shape N x K,
                where N represents size of vocab and K is the top-K nearest neighbours.
                If this is set to `None`, we have to compute nearest neighbours
                on the fly for `nearest_neighbours` method, which is costly.
        }
        @return: None
        """
        self.embedding_matrix = embedding_matrix
        self._word2index = word2index
        self._index2word = index2word
        self.nn_matrix = nn_matrix
        self._device = device or default_device

        # Dictionary for caching results
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)
        self._nn_cache = {}

    def __getitem__(self, index: Union[str, int]) -> np.ndarray:
        """Gets the embedding vector for word/id
        Args:
            index: `index` can either be word or integer representing the id of the word.
        Returns:
            vector: 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self._word2index[index]
            except KeyError:
                return None
        try:
            return self.embedding_matrix[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word: str) -> int:
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word
        Returns:
            index
        """
        return self._word2index[word]

    def index2word(self, index: int) -> str:
        """
        Convert index to corresponding word
        Args:
            index
        Returns:
            word

        """
        return self._index2word[index]

    def get_mse_dist(
        self, a: Union[str, int], b: Union[str, int], retry_with_cpu: bool = True
    ) -> float:
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: MSE (L2) distance
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            try:
                e1 = torch.from_numpy(e1).to(self._device)
                e2 = torch.from_numpy(e2).to(self._device)
            except RuntimeError:  # CUDA out of memory
                e1 = torch.from_numpy(e1).to("cpu")
                e2 = torch.from_numpy(e2).to("cpu")
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist

        return mse_dist

    def get_cos_sim(
        self, a: Union[str, int], b: Union[str, int], retry_with_cpu: bool = True
    ) -> float:
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: cosine similarity
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self._cos_sim_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            try:
                e1 = torch.from_numpy(e1).to(self._device)
                e2 = torch.from_numpy(e2).to(self._device)
            except RuntimeError:  # CUDA out of memory
                e1 = torch.from_numpy(e1).to("cpu")
                e2 = torch.from_numpy(e2).to("cpu")
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
            self._cos_sim_mat[a][b] = cos_sim
        return cos_sim

    def nearest_neighbours(
        self,
        word_or_index: Union[str, int],
        topn: int,
        return_words: bool = False,
        retry_with_cpu: bool = True,
    ) -> List[Union[int, str]]:
        """
        Get top-N nearest neighbours for a word
        Args:
            word_or_index: word or ID of the word for which we're finding the nearest neighbours
            topn: Used for specifying N nearest neighbours
        Returns:
            neighbours: List of indices of the nearest neighbours
        """
        if isinstance(word_or_index, str):
            index = self._word2index[word_or_index]
        else:
            index = word_or_index
        if self.nn_matrix is not None:
            nn = self.nn_matrix[index][1 : (topn + 1)]
        else:
            try:
                nn = self._nn_cache[index][:topn]
            except KeyError:
                nn = self._nearest_neighbours(index, topn, retry_with_cpu)
        if len(nn) < topn:
            nn = self._nearest_neighbours(index, topn, retry_with_cpu)
        if return_words:
            return [self.index2word(item) for item in nn]
        else:
            return nn

    def _nearest_neighbours(
        self, index: int, topn: int, retry_with_cpu: bool = True
    ) -> List[int]:
        """ """
        try:
            embedding = torch.from_numpy(self.embedding_matrix).to(self._device)
            vector = torch.from_numpy(self.embedding_matrix[index]).to(self._device)
            dist = torch.norm(embedding - vector, dim=1, p=None)
            # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
            try:
                nn = dist.topk(topn + 1, largest=False)[1:].tolist()
            except Exception:
                nn = dist.topk(topn + 1, largest=False).indices[1:].tolist()
        except RuntimeError:  # CUDA out of memory
            embedding = self.embedding_matrix
            vector = embedding[index]
            dist = np.linalg.norm(embedding - vector, axis=1)
            nn = np.argsort(dist)[1 : topn + 1].tolist()
        self._nn_cache[index] = nn
        return nn

    def synonyms(
        self,
        word_or_index: Union[str, int],
        topn: int,
        return_words: bool = False,
        retry_with_cpu: bool = True,
    ) -> List[Union[int, str]]:
        """ """
        return self.nearest_neighbours(
            word_or_index, topn, return_words, retry_with_cpu
        )


class GensimWordEmbedding(AbstractWordEmbedding):
    """Wraps Gensim's `models.keyedvectors` module
    (https://radimrehurek.com/gensim/models/keyedvectors.html)"""

    __name__ = "GensimWordEmbedding"

    def __init__(self, keyed_vectors: "Word2VecKeyedVectors") -> NoReturn:  # noqa: F821
        """
        keyed_vectors: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        gensim = importlib.import_module("gensim")

        if isinstance(keyed_vectors, gensim.models.keyedvectors.Word2VecKeyedVectors):
            self.keyed_vectors = keyed_vectors
        else:
            raise ValueError(
                "`keyed_vectors` argument must be a "
                "`gensim.models.keyedvectors.Word2VecKeyedVectors` object"
            )

        self.keyed_vectors.init_sims()
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)

    def __getitem__(self, index: Union[int, str]) -> np.ndarray:
        """Gets the embedding vector for word/id
        Args:
            index: `index` can either be word or integer representing the id of the word.
        Returns:
            vector: 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self.keyed_vectors.vocab.get(index).index
            except KeyError:
                return None
        try:
            return self.keyed_vectors.vectors_norm[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word: str) -> int:
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word
        Returns:
            index
        """
        vocab = self.keyed_vectors.vocab.get(word)
        if vocab is None:
            raise KeyError(word)
        return vocab.index

    def index2word(self, index: int) -> str:
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
        try:
            # this is a list, so the error would be IndexError
            return self.keyed_vectors.index2word[index]
        except IndexError:
            raise KeyError(index)

    def get_mse_dist(self, a: Union[int, str], b: Union[int, str]) -> float:
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: MSE (L2) distance
        """
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.keyed_vectors.vectors_norm[a]
            e2 = self.keyed_vectors.vectors_norm[b]
            e1 = torch.tensor(e1).to(default_device)
            e2 = torch.tensor(e2).to(default_device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist
        return mse_dist

    def get_cos_sim(self, a: Union[int, str], b: Union[int, str]) -> float:
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a: Either word or integer presenting the id of the word
            b: Either word or integer presenting the id of the word
        Returns:
            distance: cosine similarity
        """
        if not isinstance(a, str):
            a = self.keyed_vectors.index2word[a]
        if not isinstance(b, str):
            b = self.keyed_vectors.index2word[b]
        cos_sim = self.keyed_vectors.similarity(a, b)
        return cos_sim

    def nearest_neighbours(
        self, index: int, topn: int, return_words: bool = True
    ) -> List[int]:
        """
        Get top-N nearest neighbours for a word
        Args:
            index: ID of the word for which we're finding the nearest neighbours
            topn: Used for specifying N nearest neighbours
        Returns:
            neighbours: List of indices of the nearest neighbours
        """
        word = self.keyed_vectors.index2word[index]
        return [
            self.word2index(i[0])
            for i in self.keyed_vectors.similar_by_word(word, topn)
        ]


class ChineseWord2Vec(WordEmbedding):
    """
    can be downloaded from
    https://cdn.data.thunlp.org/TAADToolbox/chinese-merge-word-embedding.txt.zip

    NOTE
    ----
    这个embedding质量似乎不高
    """

    __name__ = "ChineseWord2Vec"

    def __init__(self, path: Optional[str] = None, fast: bool = True):
        """ """
        import zipfile
        import io
        import time

        self._path = path or os.path.join(
            nlp_cache_dir, "chinese-merge-word-embedding.txt.zip"
        )
        download_if_needed(
            uri="chinese-merge-word-embedding.txt.zip",
            source="aitesting",
            extract=False,
        )

        if self._path.endswith("zip"):
            zf = zipfile.ZipFile(self._path)
            f = io.TextIOWrapper(zf.open("sgns.merge.filtered.word"), encoding="utf-8")
        else:
            zf = None
            f = open(self._path, "r", encoding="utf-8")
        start = time.time()
        if fast:  # 2x faster, but much more memory consuming
            content = [line.strip().split(" ") for line in f.read().splitlines()][1:]
            super().__init__(
                embedding_matrix=np.array([line[1:] for line in content], dtype=float),
                word2index={line[0]: i for i, line in enumerate(content)},
                index2word={i: line[0] for i, line in enumerate(content)},
            )
            del content
        # id2vec = np.array([l[1:] for l in content], dtype=float)
        # word2id = {l[0]:i for i,l in enumerate(content)}
        else:
            id2vec = []
            word2id = {}
            for idx, line in enumerate(f.readlines()):
                tmp = line.strip().split(" ")
                word = tmp[0]
                embed = np.array([float(x) for x in tmp[1:]])
                if len(embed) != 300:
                    continue
                word2id[word] = len(word2id)
                id2vec.append(embed)
                print(f"processed {idx+1} lines", end="\r")
            id2vec = np.stack(id2vec)
            super().__init__(
                embedding_matrix=id2vec,
                word2index=word2id,
                index2word={v: k for k, v in word2id.items()},
            )
        print(f"costs {time.time()-start:.2f} seconds")
        f.close()
        if zf:
            zf.close()


class CounterFittedEmbedding(WordEmbedding):
    """
    can be downloaded from
    https://cdn.data.thunlp.org/TAADToolbox/counter-fitted-vectors.txt.zip
    """

    __name__ = "CounterFittedEmbedding"

    def __init__(self, path: Optional[str] = None, fast: bool = True):
        """ """
        import zipfile
        import io
        import time

        self._path = path or os.path.join(
            nlp_cache_dir, "counter-fitted-vectors.txt.zip"
        )
        download_if_needed(
            uri="counter-fitted-vectors.txt.zip",
            source="aitesting",
            extract=False,
        )

        if self._path.endswith("zip"):
            zf = zipfile.ZipFile(self._path)
            f = io.TextIOWrapper(
                zf.open(os.path.basename(self._path).replace(".zip", "")),
                encoding="utf-8",
            )
        else:
            zf = None
            f = open(self._path, "r", encoding="utf-8")
        start = time.time()
        if fast:  # 2x faster, but much more memory consuming
            content = list(
                filter(
                    lambda l: len(l) == 301,
                    [line.strip().split(" ") for line in f.readlines()],
                )
            )
            super().__init__(
                embedding_matrix=np.array([line[1:] for line in content], dtype=float),
                word2index={line[0]: i for i, line in enumerate(content)},
                index2word={i: line[0] for i, line in enumerate(content)},
            )
            del content
        # id2vec = np.array([l[1:] for l in content], dtype=float)
        # word2id = {l[0]:i for i,l in enumerate(content)}
        else:
            id2vec = []
            word2id = {}
            for idx, line in enumerate(f.readlines()):
                tmp = line.strip().split(" ")
                word = tmp[0]
                embed = np.array([float(x) for x in tmp[1:]])
                if len(embed) != 300:
                    continue
                word2id[word] = len(word2id)
                id2vec.append(embed)
                print(f"processed {idx+1} lines", end="\r")
            id2vec = np.stack(id2vec)
            super().__init__(
                embedding_matrix=id2vec,
                word2index=word2id,
                index2word={v: k for k, v in word2id.items()},
            )
        print(f"costs {time.time()-start:.2f} seconds")
        f.close()
        if zf:
            zf.close()
