# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-11-10
@LastEditTime: 2021-11-10
"""

import time
from collections import Counter
from typing import NoReturn, Tuple, Sequence, List
from numbers import Real

import numpy as np

from ...utils.constraints import StopwordModification
from ...utils.transformations import (  # noqa: F401
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
from ...Datasets.base import NLPDataset
from ...utils.attacked_text import AttackedText
from .adv_detector import AdvDetector


__all__ = [
    "FGWS",
]


class FGWS(AdvDetector):
    """

    References
    ----------
    1. Mozes M, Stenetorp P, Kleinberg B, et al. Frequency-Guided Word Substitutions for Detecting Textual Adversarial Examples[C]//Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. 2021: 171-186.
    2. https://github.com/maximilianmozes/fgws
    """

    __name__ = "FGWS"

    def __init__(
        self,
        datasets: Sequence[NLPDataset],
        percentile: Real,
        language: str = "zh",
        verbose: int = 0,
    ) -> NoReturn:
        """ """
        self._datasets = datasets
        start = time.time()
        print("loading word frequencies from datasets")
        self._word_log_freq = None
        self._load_word_log_frequencies()
        print(f"loading word frequencies cost {time.time()-start:.2f} seconds")

        self._percentile = percentile
        self.freq_thr = np.percentile(
            sorted(list(self._word_log_freq.values())), self._percentile
        )
        self.language = normalize_language(language)
        self.verbose = verbose

        start = time.time()
        print("loading stopwords and word embedding...")
        if self.language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
            embedding = ChineseWord2Vec()
            self.wes = WordEmbeddingSubstitute(embedding)
            self.wns = ChineseWordNetSubstitute()
        elif self.language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
            embedding = CounterFittedEmbedding()
            self.wes = WordEmbeddingSubstitute(embedding)
            self.wns = WordNetSubstitute()
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords and word embedding loaded in {time.time()-start:.2f} seconds")
        self.constraint = StopwordModification(stopwords)

    def _load_word_log_frequencies(self) -> NoReturn:
        """ """
        self._word_log_freq = Counter()
        for dataset in self._datasets:
            self._word_log_freq += dataset.get_word_freq(use_log=True)
        self._word_log_freq = dict(self._word_log_freq)

    def _get_word_log_freq(self, word: str) -> float:
        """ """
        return self._word_log_freq.get(word, 0)

    def reset_percentile(self, percentile: Real) -> NoReturn:
        """ """
        self._percentile = percentile
        self.freq_thr = np.percentile(
            sorted(list(self._word_log_freq.values())), self._percentile
        )

    @property
    def percentile(self) -> Real:
        return self._percentile

    def detect(self, text: str) -> Tuple[str, List[Tuple[str, str, int]]]:
        """ """
        input_text = AttackedText(language=self.language, text_input=text)
        input_freqs = np.array(
            [self._get_word_log_freq(word) for word in input_text.words]
        )
        low_freq_indices = np.where(input_freqs < self.freq_thr)[0]
        low_freq_indices = low_freq_indices[
            np.isin(
                low_freq_indices,
                list(self.constraint._get_modifiable_indices(input_text)),
            )
        ]

        replacement_meta = []
        ori_text = input_text
        for idx in low_freq_indices:
            word = input_text.words[idx]
            neighbors = self.wes._get_candidates(word)
            neighbors.extend(self.wns._get_candidates(word))
            if len(neighbors) == 0:
                continue
            neighbors = {w: self._get_word_log_freq(w) for w in set(neighbors)}
            rep = max(neighbors, key=neighbors.get)
            if self._get_word_log_freq(rep) > self._get_word_log_freq(word):
                ori_text = ori_text.replace_word_at_index(idx, rep)
                replacement_meta.append((word, rep, idx))
        return ori_text.text, replacement_meta

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "percentile",
        ] + super().extra_repr_keys()
