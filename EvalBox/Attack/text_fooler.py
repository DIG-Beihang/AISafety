# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-22
@LastEditTime: 2021-09-20

Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment

https://arxiv.org/abs/1907.11932

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
    PartOfSpeech,
    WordEmbeddingDistance,
    SentenceEncoderBase,
    MultilingualUSE,
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
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
    "TextFooler",
]


class TextFooler(Attack):
    """ """

    __name__ = "TextFooler"

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
            "min_cos_sim",
            "use_threshold",
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
            kwargs: key - default value:
                max_candidates - 50
                min_cos_sim - 0.5
                use_threshold - 0.840845057
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

        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        min_cos_sim = kwargs.get("min_cos_sim", 0.5)
        constraints.append(WordEmbeddingDistance(embedding, min_cos_sim=min_cos_sim))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        _pos_tagger = {
            LANGUAGE.ENGLISH: "nltk",
            LANGUAGE.CHINESE: "jieba",
        }[self._language]
        if self._language == LANGUAGE.ENGLISH:
            constraints.append(
                PartOfSpeech(
                    language=self._language,
                    tagger=_pos_tagger,
                    allow_verb_noun_swap=True,
                )
            )
        # constraints.append(
        #     PartOfSpeech(language=self._language, tagger=_pos_tagger, allow_verb_noun_swap=True)
        # )

        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_threshold = kwargs.get("use_threshold", 0.840845057)
        start = time.time()
        print("loading universal sentence encoder...")
        use_constraint = MultilingualUSE(
            threshold=use_threshold,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        print(f"universal sentence encoder loaded in {time.time()-start:.2f} seconds")
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = WordImportanceRanking(wir_method="delete", verbose=verbose)
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        max_candidates = kwargs.get("max_candidates", 30)
        transformation = WordEmbeddingSubstitute(
            embedding,
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
        if "min_cos_sim" in kwargs:
            for c in self.constraints:
                if isinstance(c, WordEmbeddingDistance):
                    c.min_cos_sim = kwargs.get("min_cos_sim")
        if "use_threshold" in kwargs:
            for c in self.constraints:
                if isinstance(c, (SentenceEncoderBase, MultilingualUSE)):
                    c.threshold = kwargs.get("use_threshold")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
