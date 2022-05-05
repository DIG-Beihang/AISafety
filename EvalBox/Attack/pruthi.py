# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2021-09-21

Combating Adversarial Misspellings with Robust Word Recognition

https://arxiv.org/abs/1905.11268

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
    MinWordLen,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import GreedySearch
from ...utils.transformations import (
    CompositeTransformation,
    NeighboringCharacterSubstitute,
    RandomCharacterDeletion,
    RandomCharacterInsertion,
    CharacterQWERTYSubstitute,
)
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "Pruthi",
]


class Pruthi(Attack):
    """
    An implementation of the attack used in "Combating Adversarial
    Misspellings with Robust Word Recognition", Pruthi et al., 2019.

    This attack focuses on a small number of character-level changes that simulate common typos. It combines:
        - Swapping neighboring characters
        - Deleting characters
        - Inserting characters
        - Swapping characters for adjacent keys on a QWERTY keyboard.

    https://arxiv.org/abs/1905.11268
    """

    __name__ = "Pruthi"

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
            "max_perturb_num",
            "min_word_len",
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

        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper
        transformation = CompositeTransformation(
            [
                NeighboringCharacterSubstitute(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                RandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                RandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                CharacterQWERTYSubstitute(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )

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

        # only edit words of length >= 4, edit max_num_word_swaps words.
        # note that we also are not editing the same word twice, so
        # max_num_word_swaps is really the max number of character
        # changes that can be made. The paper looks at 1 and 2 char attacks.
        max_perturb_num = kwargs.get("max_perturb_num", 1)
        constraints.append(MaxWordsPerturbed(max_num_words=max_perturb_num))

        min_word_len = kwargs.get("min_word_len", 4)
        constraints.append(MinWordLen(min_length=min_word_len))

        #
        # Goal is untargeted classification
        #
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
        if "min_word_len" in kwargs:
            for c in self.constraints:
                if isinstance(c, MinWordLen):
                    c.min_length = kwargs.get("min_word_len")
        if "max_perturb_num" in kwargs:
            for c in self.constraints:
                if isinstance(c, MaxWordsPerturbed):
                    c.max_num_words = kwargs.get("max_perturb_num")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
