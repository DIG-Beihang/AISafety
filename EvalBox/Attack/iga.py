# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-21

Natural Language Adversarial Attack and Defense in Word Level

http://arxiv.org/abs/1909.06723

状态
-----
代码完成        √
手动测试完成    √
自动测试完成    X
文档完成        √
"""

import time
from typing import NoReturn, Any, Optional

import torch

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import (
    MaxWordsPerturbed,
    WordEmbeddingDistance,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import ImprovedGeneticAlgorithm
from ...utils.transformations import WordEmbeddingSubstitute
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...utils.word_embeddings import (  # noqa: F401
    GensimWordEmbedding,
    CounterFittedEmbedding,
    ChineseWord2Vec,
)
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "IGA",
]


class IGA(Attack):
    """ """

    __name__ = "IGA"

    def __init__(
        self,
        model: Optional[NLPVictimModel] = None,
        device: Optional[torch.device] = None,
        language: str = "zh",
        **kwargs: Any,
    ) -> NoReturn:
        """
        @description: The Boundary Attack
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        self.__initialized = False
        self.__valid_params = {
            "max_candidates",
            "max_perturb_percent",
            "max_mse_dist",
            "max_iters",
            "pop_size",
            "verbose",
        }
        self._language = normalize_language(language)
        self._model = model
        if not self._model:
            start = time.time()
            if self._language == LANGUAGE.CHINESE:
                print("load default Chinese victim model...")
                self._model = VictimBERTAmazonZH()
            else:
                print("load default English victim model...")
                self._model = VictimRoBERTaSST()
            print(f"default model loaded in {time.time()-start:.2f} seconds.")
        self._device = device
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """ """
        assert set(kwargs).issubset(self.__valid_params)
        if self.__initialized:
            self.__reset_params(**kwargs)
        self.__parse_params(**kwargs)
        self.__initialized = True

    def __parse_params(self, **kwargs):
        """
        @description:
        @param {
        }
        @return:
        """
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        verbose = kwargs.get("verbose", 0)
        print("start initializing attacking parameters...")
        start = time.time()
        print("loading stopwords and word embedding...")
        if self._language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
            embedding = ChineseWord2Vec()
        elif self._language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
            embedding = CounterFittedEmbedding()
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords and word embedding loaded in {time.time()-start:.2f} seconds")

        #
        # Don't modify the stopwords
        #
        constraints = [StopwordModification()]

        #
        # Maximum words perturbed percentage of 20%
        #
        max_perturb_percent = kwargs.get("max_perturb_percent", 0.2)
        constraints.append(MaxWordsPerturbed(max_percent=max_perturb_percent))

        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        max_mse_dist = kwargs.get("max_mse_dist", 0.5)
        constraints.append(WordEmbeddingDistance(embedding, max_mse_dist=max_mse_dist))

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)

        #
        # Perform word substitution with an improved genetic algorithm.
        # Fix the hyperparameter values to S = 60, M = 20, λ = 5."
        #
        pop_size = kwargs.get("pop_size", 60)
        max_iters = kwargs.get("max_iters", 20)
        search_method = ImprovedGeneticAlgorithm(
            pop_size=pop_size,
            max_iters=max_iters,
            max_replace_times_per_index=5,
            post_crossover_check=False,
        )

        #
        # Swap words with their embedding nearest-neighbors.
        # Embedding: Counter-fitted Paragram Embeddings.
        # Fix the hyperparameter value to N = Unrestricted (50)."
        #
        max_candidates = kwargs.get("max_candidates", 50)
        transformation = WordEmbeddingSubstitute(
            embedding=embedding,
            max_candidates=max_candidates,
            verbose=verbose,
        )

        super().__init__(
            model=self._model,
            device=self._device,
            IsTargeted=False,
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method,
            language=self._language,
            verbose=verbose,
        )

    def __reset_params(self, **kwargs: Any) -> NoReturn:
        """ """
        if "max_candidates" in kwargs:
            self.transformation.max_candidates = kwargs.get("max_candidates")
        if "max_perturb_percent" in kwargs:
            for c in self.constraints:
                if isinstance(c, MaxWordsPerturbed):
                    c.max_percent = kwargs.get("max_perturb_percent")
        if "max_mse_dist" in kwargs:
            for c in self.constraints:
                if isinstance(c, WordEmbeddingDistance):
                    c.max_mse_dist = kwargs.get("max_mse_dist")
        if "pop_size" in kwargs:
            self.search_method.pop_size = kwargs.get("pop_size")
        if "max_iters" in kwargs:
            self.search_method.max_iters = kwargs.get("max_iters")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
