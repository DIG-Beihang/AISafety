# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-20

Contextualized Perturbation for Textual Adversarial Attack

https://arxiv.org/abs/2009.07502

状态
-----
代码完成        √
手动测试完成    X  # CUDA out of memory
自动测试完成    X
文档完成        √
"""

import os
import time
from typing import NoReturn, Any, Optional

import torch
import transformers

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import (
    SentenceEncoderBase,
    MultilingualUSE,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import GreedySearch
from ...utils.transformations import (
    CompositeTransformation,
    WordMaskedLMInsertion,
    WordMaskedLMMerge,
    WordMaskedLMSubstitute,
)
from ...utils.misc import nlp_cache_dir
from ...utils._download_data import download_if_needed
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "CLARE",
]


class CLARE(Attack):
    """ """

    __name__ = "CLARE"

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
        # "This paper presents CLARE, a ContextuaLized AdversaRial Example generation model
        # that produces fluent and grammatical outputs through a mask-then-infill procedure.
        # CLARE builds on a pre-trained masked language model and modifies the inputs in a context-aware manner.
        # We propose three contex-tualized  perturbations, Replace, Insert and Merge, allowing for generating outputs of
        # varied lengths."
        #
        # "We  experiment  with  a  distilled  version  of RoBERTa (RoBERTa_{distill}; Sanh et al., 2019)
        # as the masked language model for contextualized infilling."
        # Because BAE and CLARE both use similar replacement papers, we use BAE's replacement method here.

        verbose = kwargs.get("verbose", 0)

        if self._language == LANGUAGE.ENGLISH:
            path = os.path.join(nlp_cache_dir, "distilroberta-base")
            if not os.path.exists(path):
                # raise ValueError("暂不支持在线加载模型")
                path = download_if_needed(
                    uri="distilroberta-base",
                    source="aitesting",
                )
            shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(path)
            shared_tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        elif self._language == LANGUAGE.CHINESE:
            path = os.path.join(nlp_cache_dir, "bert-base-chinese")
            if not os.path.exists(path):
                # raise ValueError("暂不支持在线加载模型")
                path = download_if_needed(
                    uri="bert-base-chinese",
                    source="aitesting",
                )
            shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(path)
            shared_tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        max_candidates = kwargs.get("max_candidates", 50)
        transformation = CompositeTransformation(
            [
                WordMaskedLMSubstitute(
                    language=self._language,
                    method="bae",
                    masked_lm_or_path=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=5e-4,
                ),
                WordMaskedLMInsertion(
                    language=self._language,
                    masked_lm_or_path=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=0.0,
                ),
                WordMaskedLMMerge(
                    language=self._language,
                    masked_lm_or_path=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=5e-3,
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

        # "A  common  choice  of sim(·,·) is to encode sentences using neural networks,
        # and calculate their cosine similarity in the embedding space (Jin et al., 2020)."
        # The original implementation uses similarity of 0.7.
        use_threshold = kwargs.get("use_threshold", 0.7)
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
        # "To achieve this,  we iteratively apply the actions,
        #  and first select those minimizing the probability of outputting the gold label y from f."
        #
        # "Only one of the three actions can be applied at each position, and we select the one with the highest score."
        #
        # "Actions are iteratively applied to the input, until an adversarial example is found or a limit of actions T
        # is reached.
        #  Each step selects the highest-scoring action from the remaining ones."
        #
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
