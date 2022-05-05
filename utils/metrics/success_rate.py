# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2021-09-22

基于攻击成功率的 metric，
基于 TextAttack 的 AttackSuccessRate
"""

from typing import NoReturn, Union, Sequence

from .base import Metric
from ..strings import normalize_language, LANGUAGE
from ...EvalBox.Attack.attack_result import (
    GoalFunctionResult,
    FailedAttackResult,
    SkippedAttackResult,
)


__all__ = [
    "SuccessRate",
]


class SuccessRate(Metric):
    """
    Calculates all metrics related to number of succesful, failed and skipped results in an attack
    """

    __name__ = "SuccessRate"

    def __init__(self, language: Union[str, LANGUAGE]) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self.failed_attacks = 0
        self.skipped_attacks = 0
        self.successful_attacks = 0

        self.all_metrics = {}

    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        """ """
        self.results = results
        self.total_attacks = len(self.results)

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                self.failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                self.skipped_attacks += 1
                continue
            else:
                self.successful_attacks += 1

        # Calculated numbers
        self.all_metrics["successful_attacks"] = self.successful_attacks
        self.all_metrics["failed_attacks"] = self.failed_attacks
        self.all_metrics["skipped_attacks"] = self.skipped_attacks

        # Percentages wrt the calculations
        self.all_metrics["original_accuracy"] = self.__original_accuracy_perc()
        self.all_metrics["attack_accuracy_perc"] = self.__attack_accuracy_perc()
        self.all_metrics["attack_success_rate"] = self.__attack_success_rate_perc()

        return self.all_metrics

    def __original_accuracy_perc(self) -> float:
        """ """
        original_accuracy = (
            (self.total_attacks - self.skipped_attacks) * 100.0 / (self.total_attacks)
        )
        original_accuracy = round(original_accuracy, 2)
        return original_accuracy

    def __attack_accuracy_perc(self) -> float:
        """ """
        accuracy_under_attack = (self.failed_attacks) * 100.0 / (self.total_attacks)
        accuracy_under_attack = round(accuracy_under_attack, 2)
        return accuracy_under_attack

    def __attack_success_rate_perc(self) -> float:
        """ """
        if self.successful_attacks + self.failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                self.successful_attacks
                * 100.0
                / (self.successful_attacks + self.failed_attacks)
            )
        attack_success_rate = round(attack_success_rate, 2)
        return attack_success_rate
