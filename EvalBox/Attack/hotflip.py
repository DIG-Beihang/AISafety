# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-13

HotFlip: White-Box Adversarial Examples for Text Classification

https://arxiv.org/abs/1712.06751

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
from ...utils.constraints import (  # noqa: F401
    PartOfSpeech,
    RepeatModification,
    StopwordModification,
    MaxWordsPerturbed,
    WordEmbeddingDistance,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import BeamSearch
from ...utils.transformations import WordGradientSubstitute
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
    "HotFlip",
]


class HotFlip(Attack):
    """ """

    __name__ = "HotFlip"

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
            "top_n",
            "max_num_words",
            "min_cos_sim",
            "beam_width",
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
        # fmt: on
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # 0. "We were able to create only 41 examples (2% of the correctly-
        # classified instances of the SST test set) with one or two flips."
        #
        max_num_words = kwargs.get("max_num_words", 2)
        constraints.append(MaxWordsPerturbed(max_num_words=max_num_words))
        #
        # 1. "The cosine similarity between the embedding of words is bigger than a
        #   threshold (0.8)."
        #
        min_cos_sim = kwargs.get("min_cos_sim", 0.8)
        constraints.append(WordEmbeddingDistance(embedding, min_cos_sim=min_cos_sim))
        #
        # 2. "The two words have the same part-of-speech."
        #
        _pos_tagger = {
            LANGUAGE.ENGLISH: "nltk",
            LANGUAGE.CHINESE: "jieba",
        }[self._language]
        if self._language == LANGUAGE.ENGLISH:
            # 中文暂时不添加POS限制
            constraints.append(
                PartOfSpeech(language=self._language, tagger=_pos_tagger)
            )
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        beam_width = kwargs.get("beam_width", 10)
        search_method = BeamSearch(beam_width=beam_width, verbose=verbose)

        top_n = kwargs.get("top_n", 1)
        transformation = WordGradientSubstitute(
            self._language, self._model, top_n=top_n, verbose=verbose
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
        if "top_n" in kwargs:
            self.transformation.top_n = kwargs.get("top_n")
        if "beam_width" in kwargs:
            self.search_method.beam_width = kwargs.get("beam_width")
        if "min_cos_sim" in kwargs:
            for c in self.constraints:
                if isinstance(c, WordEmbeddingDistance):
                    c.min_cos_sim = kwargs.get("min_cos_sim")
        if "max_num_words" in kwargs:
            for c in self.constraints:
                if isinstance(c, (MaxWordsPerturbed)):
                    c.max_num_words = kwargs.get("max_num_words")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
