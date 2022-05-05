# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-21

BERT-ATTACK: Adversarial Attack Against BERT Using BERT

https://arxiv.org/abs/2004.09984

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
    "BERTAttack",
]


class BERTAttack(Attack):
    """ """

    __name__ = "BERTAttack"

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
            "max_perturb_percent",
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

        # "We only take ε percent of the most important words since we tend to keep
        # perturbations minimum."
        #
        # [from correspondence with the author]
        # "Word percentage allowed to change is set to 0.4 for most data-sets, this
        # parameter is trivial since most attacks only need a few changes. This
        # epsilon is only used to avoid too much queries on those very hard samples."
        max_perturb_percent = kwargs.get("max_perturb_percent", 0.4)
        constraints.append(MaxWordsPerturbed(max_percent=max_perturb_percent))

        # "As used in TextFooler (Jin et al., 2019), we also use Universal Sentence
        # Encoder (Cer et al., 2018) to measure the semantic consistency between the
        # adversarial sample and the original sequence. To balance between semantic
        # preservation and attack success rate, we set up a threshold of semantic
        # similarity score to filter the less similar examples."
        #
        # [from correspondence with author]
        # "Over the full texts, after generating all the adversarial samples, we filter
        # out low USE score samples. Thus the success rate is lower but the USE score
        # can be higher. (actually USE score is not a golden metric, so we simply
        # measure the USE score over the final texts for a comparison with TextFooler).
        # For datasets like IMDB, we set a higher threshold between 0.4-0.7; for
        # datasets like MNLI, we set threshold between 0-0.2."
        #
        # Since the threshold in the real world can't be determined from the training
        # data, the TextAttack implementation uses a fixed threshold - determined to
        # be 0.2 to be most fair.
        use_threshold = kwargs.get("use_threshold", 0.2)
        start = time.time()
        print("loading universal sentence encoder...")
        use_constraint = MultilingualUSE(
            threshold=use_threshold,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        print(f"universal sentence encoder loaded in {time.time()-start:.2f} seconds")
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)
        #
        # "We first select the words in the sequence which have a high significance
        # influence on the final output logit. Let S = [w0, ··· , wi ··· ] denote
        # the input sentence, and oy(S) denote the logit output by the target model
        # for correct label y, the importance score Iwi is defined as
        # Iwi = oy(S) − oy(S\wi), where S\wi = [w0, ··· , wi−1, [MASK], wi+1, ···]
        # is the sentence after replacing wi with [MASK]. Then we rank all the words
        # according to the ranking score Iwi in descending order to create word list
        # L."
        search_method = WordImportanceRanking(wir_method="unk", verbose=verbose)
        # [from correspondence with the author]
        # Candidate size K is set to 48 for all data-sets.
        max_candidates = kwargs.get("max_candidates", 48)
        transformation = WordMaskedLMSubstitute(
            language=self._language,
            method="bert-attack",
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
        if "max_perturb_percent" in kwargs:
            for c in self.constraints:
                if isinstance(c, MaxWordsPerturbed):
                    c.max_percent = kwargs.get("max_perturb_percent")
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
