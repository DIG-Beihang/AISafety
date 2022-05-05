# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2022-03-19

Beyond Accuracy: Behavioral Testing of NLP models with CheckList

https://arxiv.org/abs/2005.04118

状态
-----
代码完成        √
手动测试完成    X  问题：flair pos tag
自动测试完成    X
文档完成        √
"""

import time
from typing import NoReturn, Any, Optional

import torch

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import RepeatModification
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import GreedySearch
from ...utils.transformations import (  # noqa: F401
    CompositeTransformation,
    WordChangeLocSubstitute,
    WordChangeNameSubstitute,
    WordChangeNumSubstitute,
    WordExtendSubstitute,
    WordContractSubstitute,
    ChineseWordNetSubstitute,
    ChineseCiLinSubstitute,
)
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "CheckList",
]


class CheckList(Attack):
    """An implementation of the attack used in "Beyond Accuracy: Behavioral
    Testing of NLP models with CheckList", Ribeiro et al., 2020.

    This attack focuses on a number of attacks used in the Invariance Testing
    Method: Contraction, Extension, Changing Names, Number, Location

    https://arxiv.org/abs/2005.04118
    """

    __name__ = "CheckList"

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
        verbose = kwargs.get("verbose", 0)

        if self._language == LANGUAGE.ENGLISH:
            transformation = CompositeTransformation(
                [
                    WordExtendSubstitute(),
                    WordContractSubstitute(),
                    WordChangeNameSubstitute(),
                    WordChangeNumSubstitute(),
                    WordChangeLocSubstitute(),
                ]
            )
        else:
            # transformation = ChineseWordNetSubstitute()
            transformation = ChineseCiLinSubstitute()

        # Need this constraint to prevent extend and contract modifying each others' changes and forming infinite loop
        constraints = [RepeatModification()]

        # Untargeted attack & GreedySearch
        goal_function = UntargetedClassification(self._model)
        search_method = GreedySearch(verbose=verbose)

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
