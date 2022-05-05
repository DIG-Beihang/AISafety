# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2022-03-05

基于编辑距离的 metric，
基于 OpenAttack 的 metric.levenshtein.py
"""

from typing import NoReturn, Union, Sequence

import numpy as np

from .base import Metric
from ..strings import normalize_language, LANGUAGE

# from ...utils.goal_functions import GoalFunctionResult
from ...EvalBox.Attack.attack_result import (
    AttackResult,
    FailedAttackResult,
    SkippedAttackResult,
)


__all__ = [
    "EditDistanceMetric",
]


class EditDistanceMetric(Metric):
    """ """

    __name__ = "EditDistanceMetric"

    def __init__(self, language: Union[str, LANGUAGE]) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self.all_metrics = {}

    def calculate(self, results: Sequence[AttackResult]) -> dict:
        """ """
        words_edit_distance = []
        char_edit_distance = []
        for i, result in enumerate(self.results):
            self.all_num_words[i] = len(result.original_result.attacked_text.words)

            if isinstance(result, FailedAttackResult) or isinstance(
                result, SkippedAttackResult
            ):
                continue

            words_edit_distance.append(
                EditDistanceMetric._edit_distance(
                    result.original_result.attacked_text.words,
                    result.perturbed_result.attacked_text.words,
                )
            )

            char_edit_distance.append(
                EditDistanceMetric._edit_distance(
                    result.original_result.attacked_text.text,
                    result.perturbed_result.attacked_text.text,
                )
            )

        self.all_metrics["words_edit_distance"] = round(np.mean(words_edit_distance), 2)
        self.all_metrics["char_edit_distance"] = round(np.mean(char_edit_distance), 2)

        return self.all_metrics

    @staticmethod
    def _edit_distance(
        a: Union[str, Sequence[str]], b: Union[str, Sequence[str]]
    ) -> int:
        """ """
        la = len(a)
        lb = len(b)
        f = np.zeros((la + 1, lb + 1), dtype=np.uint64)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
        return f[la][lb]
