# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-25
@LastEditTime: 2021-12-23

Towards Improving Adversarial Training of NLP Models.

https://arxiv.org/abs/2109.00544

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
from ...utils.constraints import (  # noqa: F401
    PartOfSpeech,
    WordEmbeddingDistance,
    SentenceEncoderBase,
    MultilingualUSE,  # to add `BERT`
    InputColumnModification,
    RepeatModification,
    StopwordModification,
    MaxModificationRate,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import WordEmbeddingSubstitute, WordMaskedLMSubstitute
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
    "A2T",
]


class A2T(Attack):
    """ """

    __name__ = "A2T"

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
            "max_modification_rate",
            "mlm",
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
                    allow_verb_noun_swap=False,
                )
            )
        # constraints.append(
        #     PartOfSpeech(language=self._language, tagger=_pos_tagger, allow_verb_noun_swap=True)
        # )

        use_threshold = kwargs.get("use_threshold", 0.9)
        start = time.time()
        print("loading universal sentence encoder...")
        # TODO: replace with `BERT` as in the original code/paper
        use_constraint = MultilingualUSE(
            threshold=use_threshold,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        print(f"universal sentence encoder loaded in {time.time()-start:.2f} seconds")
        constraints.append(use_constraint)

        max_modification_rate = kwargs.get("max_modification_rate", 0.1)
        constraints.append(
            MaxModificationRate(max_rate=max_modification_rate, min_threshold=4)
        )

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = WordImportanceRanking(wir_method="gradient", verbose=verbose)

        mlm = kwargs.get("mlm", False)
        min_cos_sim = kwargs.get("min_cos_sim", 0.8)
        max_candidates = kwargs.get("max_candidates", 20)
        if mlm:
            transformation = WordMaskedLMSubstitute(
                method="bae",
                max_candidates=max_candidates,
                min_confidence=0.0,
                batch_size=16,
                verbose=verbose,
            )
        else:
            transformation = WordEmbeddingSubstitute(
                embedding,
                max_candidates=max_candidates,
                verbose=verbose,
            )
            constraints.append(
                WordEmbeddingDistance(embedding, min_cos_sim=min_cos_sim)
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
        if "mlm" in kwargs:
            # TODO: reset transformation and (possibly) add or delete contraints
            pass
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
        if "max_modification_rate" in kwargs:
            for c in self.constraints:
                if isinstance(c, MaxModificationRate):
                    c.max_rate = kwargs.get("max_modification_rate")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
