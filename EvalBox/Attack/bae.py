# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-21

BAE: BERT-based Adversarial Examples for Text Classification

https://arxiv.org/pdf/2004.01970

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
    SentenceEncoderBase,
    MultilingualUSE,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import WordMaskedLMSubstitute
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "BAE",
]


class BAE(Attack):
    """
    4 attack modes for BAE based on the
        R and I operations, where for each token t in S:
        • BAE-R: Replace token t (See Algorithm 1)
        • BAE-I: Insert a token to the left or right of t
        • BAE-R/I: Either replace token t or insert a
        token to the left or right of t
        • BAE-R+I: First replace token t, then insert a
        token to the left or right of t
    """

    __name__ = "BAE"

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

        # "To ensure semantic similarity on introducing perturbations in the input
        # text, we filter the set of top-K masked tokens (K is a pre-defined
        # constant) predicted by BERT-MLM using a Universal Sentence Encoder (USE)
        # (Cer et al., 2018)-based sentence similarity scorer."
        #
        # "[We] set a threshold of 0.8 for the cosine similarity between USE-based
        # embeddings of the adversarial and input text."
        #
        # [from email correspondence with the author]
        # "For a fair comparison of the benefits of using a BERT-MLM in our paper,
        # we retained the majority of TextFooler's specifications. Thus we:
        # 1. Use the USE for comparison within a window of size 15 around the word
        # being replaced/inserted.
        # 2. Set the similarity score threshold to 0.1 for inputs shorter than the
        # window size (this translates roughly to almost always accepting the new text).
        # 3. Perform the USE similarity thresholding of 0.8 with respect to the text
        # just before the replacement/insertion and not the original text (For
        # example: at the 3rd R/I operation, we compute the USE score on a window
        # of size 15 of the text obtained after the first 2 R/I operations and not
        # the original text).
        # ...
        # To address point (3) from above, compare the USE with the original text
        # at each iteration instead of the current one (While doing this change
        # for the R-operation is trivial, doing it for the I-operation with the
        # window based USE comparison might be more involved)."
        #
        # Finally, since the BAE code is based on the TextFooler code, we need to
        # adjust the threshold to account for the missing / pi in the cosine
        # similarity comparison. So the final threshold is 1 - (1 - 0.8) / pi
        # = 1 - (0.2 / pi) = 0.936338023.
        use_threshold = kwargs.get("use_threshold", 0.936338023)
        start = time.time()
        print("loading universal sentence encoder...")
        use_constraint = MultilingualUSE(
            threshold=use_threshold,
            metric="cosine",
            compare_against_original=True,
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

        # "In this paper, we present a simple yet novel technique: BAE (BERT-based
        # Adversarial Examples), which uses a language model (LM) for token
        # replacement to best fit the overall context. We perturb an input sentence
        # by either replacing a token or inserting a new token in the sentence, by
        # means of masking a part of the input and using a LM to fill in the mask."
        #
        # We only consider the top K=50 synonyms from the MLM predictions.
        #
        # [from email correspondance with the author]
        # "When choosing the top-K candidates from the BERT masked LM, we filter out
        # the sub-words and only retain the whole words (by checking if they are
        # present in the GloVE vocabulary)"
        #
        max_candidates = kwargs.get("max_candidates", 50)
        transformation = WordMaskedLMSubstitute(
            language=self._language,
            method="bae",
            max_candidates=max_candidates,
            min_confidence=0.0,
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
