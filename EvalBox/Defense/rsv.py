# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-11-10
@LastEditTime: 2021-11-10
"""

import random
import time
from typing import NoReturn, Tuple, List

import torch

from ..Attack.attack_generator import AdvSampleGenerator
from ...Models.base import NLPVictimModel
from ...utils.constraints import (
    RepeatModification,
    StopwordModification,
)
from ...utils.transformations import (
    WordEmbeddingSubstitute,
    WordNetSubstitute,
    ChineseWordNetSubstitute,
    RandomCompositeTransformation,
)
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...utils.word_embeddings import (  # noqa: F401
    GensimWordEmbedding,
    CounterFittedEmbedding,
    ChineseWord2Vec,
)
from ...utils.attacked_text import AttackedText
from .adv_detector import AdvDetector


__all__ = [
    "RSV",
]


class RSV(AdvDetector):
    """

    References:
    -----------
    Wang X, Xiong Y, He K. Randomized Substitution and Vote for Textual Adversarial Example Detection[J]. arXiv preprint arXiv:2109.05698, 2021

    Example
    -------
    ```python
    from text.Models.TestModel.roberta_sst import VictimRoBERTaSST
    from text.EvalBox.Defense.rsv import RSV

    adv_detector = RSV(10, VictimRoBERTaSST(), "en")
    adv_example = "The Rock is made to be the 21st Century's new `` Conan '' and that he's supposed to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
    # note that this adversarial example is generated from an example from `SST` using attack recipe `BAE`
    # "The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."

    adv_detector.detect(adv_example)
    # output is `(False, 0)`, hence failed
    ```
    """

    def __init__(
        self,
        n_votes: int,
        tgt_clf: NLPVictimModel,
        language: str = "zh",
        verbose: int = 0,
    ) -> NoReturn:
        """ """
        self.n_votes = n_votes
        self.tgt_clf = tgt_clf
        self.language = normalize_language(language)

        start = time.time()
        print("loading stopwords and word embedding...")
        if self.language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
            embedding = ChineseWord2Vec()
            wns_cls = ChineseWordNetSubstitute
        elif self.language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
            embedding = CounterFittedEmbedding()
            wns_cls = WordNetSubstitute
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords and word embedding loaded in {time.time()-start:.2f} seconds")
        self.verbose = verbose

        transformation = RandomCompositeTransformation(
            [
                WordEmbeddingSubstitute(
                    embedding,
                    max_candidates=n_votes,
                    verbose=verbose,
                ),
                wns_cls(),
            ]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        self.adv_gen = AdvSampleGenerator(
            transformation=transformation,
            constraints=constraints,
            language=language,
            verbose=verbose,
        )

    def detect(self, text: str) -> Tuple[bool, int]:
        """ """
        ori_text = AttackedText(language=self.language, text_input=text)
        adv_samples = [t.text for t in self.adv_gen.get_transformations(ori_text)]
        if len(adv_samples) > self.n_votes:
            adv_samples = random.sample(adv_samples, self.n_votes)

        adv_out = torch.argmax(
            torch.mean(torch.softmax(self.tgt_clf(adv_samples), dim=1), dim=0)
        ).item()
        ori_out = torch.argmax(
            torch.mean(torch.softmax(self.tgt_clf([text]), dim=1), dim=0)
        ).item()
        if adv_out == ori_out:
            return False, adv_out
        else:
            return True, adv_out

    def extra_repr_keys() -> List[str]:
        """ """
        return [
            "n_votes",
        ] + super().extra_repr_keys()
