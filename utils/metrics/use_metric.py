# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2021-09-22

基于 universal sentence encoder 得到的距离的 metric，
基于 TextAttack 的 USEMetric，参考了 OpenAttack 的 usencoder.py
"""

from typing import NoReturn, Union, Any, Sequence

from .base import Metric
from ..strings import LANGUAGE
from ..constraints import (  # noqa: F401
    SentenceEncoderBase,
    MultilingualUSE,
    ThoughtVector,
)
from ...EvalBox.Attack.attack_result import (  # noqa: F401
    GoalFunctionResult,
    FailedAttackResult,
    SkippedAttackResult,
)


__all__ = [
    "UniversalSentenceEncoderMetric",
]


class UniversalSentenceEncoderMetric(Metric):
    """
    Calculates average USE similarity on all successfull attacks
    """

    __name__ = "UniversalSentenceEncoderMetric"

    def __init__(self, language: Union[str, LANGUAGE], **kwargs: Any) -> NoReturn:
        """ """
        self.use_obj = MultilingualUSE()
        self.use_obj.model = MultilingualUSE()
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        """ """
        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(result.original_result.attacked_text)
                self.successful_candidates.append(result.perturbed_result.attacked_text)

        use_scores = []
        for c in range(len(self.original_candidates)):
            use_scores.append(
                self.use_obj._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                ).item()
            )

        self.all_metrics["avg_attack_use_score"] = round(
            sum(use_scores) / len(use_scores), 2
        )

        return self.all_metrics
