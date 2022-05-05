# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-13

Generating Natural Langauge Adversarial Examples through Probability Weighted Word Saliency

https://www.aclweb.org/anthology/P19-1103/

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
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import WordNetSubstitute, ChineseWordNetSubstitute
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "PWWS",
]


class PWWS(Attack):
    """ """

    __name__ = "PWWS"

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
        # Don't modify the same word twice or the stopwords
        #
        # fmt: off
        verbose = kwargs.get("verbose", 0)
        print("start initializing attacking parameters...")
        start = time.time()
        print("loading stopwords...")
        if self._language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
            transformation = ChineseWordNetSubstitute()
        elif self._language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
            transformation = WordNetSubstitute()
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords loaded in {time.time()-start:.2f} seconds")
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = WordImportanceRanking(
            wir_method="weighted-saliency", verbose=verbose
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
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
