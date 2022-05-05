# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2022-01-05
@LastEditTime: 2022-01-05

Pathologies of Neural Models Make Interpretations Difficult.

这个对抗攻击方法可能不能用于攻击分类模型，待确认。

https://arxiv.org/abs/1804.07781

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
from ...utils.constraints import (
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import InputReduction as IR
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import WordDeletion
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "InputReduction",
]


class InputReduction(Attack):
    """ """

    __name__ = "InputReduction"

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
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        verbose = kwargs.get("verbose", 0)
        print("start initializing attacking parameters...")
        start = time.time()
        print("loading stopwords...")
        if self._language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
        elif self._language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords loaded in {time.time()-start:.2f} seconds")

        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        #
        # Goal is untargeted classification
        #
        goal_function = IR(self._model, maximizable=True)
        #
        # "For each word in an input sentence, we measure its importance by the
        # change in the confidence of the original prediction when we remove
        # that word from the sentence."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = WordImportanceRanking(wir_method="delete", verbose=verbose)

        transformation = WordDeletion()

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
