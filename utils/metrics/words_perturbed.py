# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2021-09-22

基于词替换数量的 metric，
基于 TextAttack 的 WordsPerturbed
"""

from typing import NoReturn, Union, Sequence

import numpy as np

from .base import Metric
from ..strings import normalize_language, LANGUAGE
from ...EvalBox.Attack.attack_result import (  # noqa: F401
    GoalFunctionResult,
    FailedAttackResult,
    SkippedAttackResult,
)


__all__ = [
    "WordsPerturbedMetric",
]


class WordsPerturbedMetric(Metric):
    """ """

    __name__ = "WordsPerturbedMetric"

    def __init__(self, language: Union[str, LANGUAGE]) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self.total_attacks = 0
        self.all_num_words = None
        self.perturbed_word_percentages = None
        self.num_words_changed_until_success = 0
        self.all_metrics = {}

    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        """ """
        self.results = results
        self.total_attacks = len(self.results)
        self.all_num_words = np.zeros(len(self.results))
        self.perturbed_word_percentages = np.zeros(len(self.results))
        self.num_words_changed_until_success = np.zeros(2**16)
        self.max_words_changed = 0

        for i, result in enumerate(self.results):
            self.all_num_words[i] = len(result.original_result.attacked_text.words)

            if isinstance(result, FailedAttackResult) or isinstance(
                result, SkippedAttackResult
            ):
                continue

            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            self.num_words_changed_until_success[num_words_changed - 1] += 1
            self.max_words_changed = max(
                self.max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0

            self.perturbed_word_percentages[i] = perturbed_word_percentage

        self.all_metrics["avg_word_perturbed"] = self.__avg_number_word_perturbed_num()
        self.all_metrics["avg_word_perturbed_perc"] = self.__avg_perturbation_perc()
        self.all_metrics["max_words_changed"] = self.max_words_changed
        self.all_metrics[
            "num_words_changed_until_success"
        ] = self.num_words_changed_until_success

        return self.all_metrics

    def __avg_number_word_perturbed_num(self) -> float:
        """ """
        average_num_words = self.all_num_words.mean()
        average_num_words = round(average_num_words, 2)
        return average_num_words

    def __avg_perturbation_perc(self) -> float:
        """ """
        self.perturbed_word_percentages = self.perturbed_word_percentages[
            self.perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = self.perturbed_word_percentages.mean()
        average_perc_words_perturbed = round(average_perc_words_perturbed, 2)
        return average_perc_words_perturbed
