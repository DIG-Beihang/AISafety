# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-21

Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers

https://arxiv.org/abs/1801.04354

状态
-----
代码完成        √
手动测试完成    √
自动测试完成    √
文档完成        √
"""

import time
from typing import NoReturn, Any, Optional

import torch

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import (
    EditDistance,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import (
    CompositeTransformation,
    NeighboringCharacterSubstitute,
    RandomCharacterDeletion,
    RandomCharacterInsertion,
    RandomCharacterSubstitute,
)
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "DeepWordBug",
]


class DeepWordBug(Attack):
    """ """

    __name__ = "DeepWordBug"

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
            "use_all_transformations",
            "max_edit_distance",
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

        self.use_all_transformations = kwargs.get("use_all_transformations", True)

        #
        # Swap characters out from words. Choose the best of four potential transformations.
        #
        if self.use_all_transformations:
            # We propose four similar methods:
            transformation = CompositeTransformation(
                [
                    # (1) Swap: Swap two adjacent letters in the word.
                    NeighboringCharacterSubstitute(),
                    # (2) Substitution: Substitute a letter in the word with a random letter.
                    RandomCharacterSubstitute(),
                    # (3) Deletion: Delete a random letter from the word.
                    RandomCharacterDeletion(),
                    # (4) Insertion: Insert a random letter in the word.
                    RandomCharacterInsertion(),
                ]
            )
        else:
            # We use the Combined Score and the Substitution Transformer to generate
            # adversarial samples, with the maximum edit distance difference of 30
            # (ϵ = 30).
            transformation = RandomCharacterSubstitute()

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

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        max_edit_distance = kwargs.get("max_edit_distance", 30)
        constraints.append(EditDistance(max_edit_distance=max_edit_distance))

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        # "To achieve this,  we iteratively apply the actions,
        #  and first select those minimizing the probability of outputting the gold label y from f."
        #
        # "Only one of the three actions can be applied at each position, and we select the one with the highest score."
        #
        # "Actions are iteratively applied to the input, until an adversarial example is found or a limit of actions T
        # is reached.
        #  Each step selects the highest-scoring action from the remaining ones."
        #
        # TODO: should one set wir_method="delete" for WordImportanceRanking
        search_method = WordImportanceRanking(verbose=verbose)

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
        if "max_edit_distance" in kwargs:
            for c in self.constraints:
                if isinstance(c, EditDistance):
                    c.max_edit_distance = kwargs.get("max_edit_distance")
        self.use_all_transformations = kwargs.get(
            "use_all_transformations", self.use_all_transformations
        )
        # TODO: adjust transformations according to self.use_all_transformations
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
