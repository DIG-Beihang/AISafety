# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2022-01-05
@LastEditTime: 2022-03-19

Adversarial Examples for Natural Language Classification Problems.

https://openreview.net/forum?id=r1QZ3zbAZ

状态
-----
代码完成        √
手动测试完成    √
自动测试完成    X
文档完成        X
"""

import time
from typing import NoReturn, Any, Optional

import torch

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import (  # noqa: F401
    GPT2,
    GoogleLanguageModel,
    Learning2Write,
    ThoughtVector,
    MaxWordsPerturbed,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import GreedySearch
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
    "Kuleshov",
]


class Kuleshov(Attack):
    """ """

    __name__ = "Kuleshov"

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
            "verbose",
            "max_candidates",
            "max_perturb_percent",
            "thought_vector_thr",
            "gpt2_thr",
            "target_max_score",
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
        # fmt: off
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

        max_candidates = kwargs.get("max_candidates", 15)
        transformation = WordEmbeddingSubstitute(
            embedding, max_candidates=max_candidates, verbose=verbose,
        )

        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # Maximum of 50% of words perturbed (δ in the paper).
        #
        max_perturb_percent = kwargs.get("max_perturb_percent", 0.5)
        constraints.append(MaxWordsPerturbed(max_percent=max_perturb_percent))
        #
        # Maximum thought vector Euclidean distance of λ_1 = 0.2. (eq. 4)
        #
        thought_vector_thr = kwargs.get("thought_vector_thr", 0.2)
        constraints.append(
            ThoughtVector(
                self._language,
                embedding,
                threshold=thought_vector_thr,
                metric="max_euclidean",
            )
        )
        #
        #
        # Maximum language model log-probability difference of λ_2 = 2. (eq. 5)
        #
        gpt2_thr = kwargs.get("gpt2_thr", 2.0)
        constraints.append(GPT2(max_log_prob_diff=2.0, language=self._language))

        #
        # Goal is untargeted classification
        #
        target_max_score = kwargs.get("target_max_score", 0.7)
        goal_function = UntargetedClassification(
            self._model, target_max_score=target_max_score
        )
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = GreedySearch()

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
                    c.min_cos_sim = kwargs.get("max_perturb_percent")
        if "thought_vector_thr" in kwargs:
            for c in self.constraints:
                if isinstance(c, ThoughtVector):
                    c.threshold = kwargs.get("thought_vector_thr")
        if "gpt2_thr" in kwargs:
            for c in self.constraints:
                if isinstance(c, GPT2):
                    c.threshold = kwargs.get("gpt2_thr")
        if "target_max_score" in kwargs:
            self.goal_function.target_max_score = kwargs.get("target_max_score")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
