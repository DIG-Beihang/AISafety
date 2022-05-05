# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2021-09-22

基于进行变换数量的 metric，
基于 TextAttack 的 AttackQueries
"""

from typing import NoReturn, Union, Sequence

import numpy as np

from .base import Metric
from ..strings import normalize_language, LANGUAGE
from ...EvalBox.Attack.attack_result import (
    GoalFunctionResult,
    SkippedAttackResult,
)


__all__ = [
    "NumQueriesMetric",
]


class NumQueriesMetric(Metric):
    """
    Calculates all metrics related to number of queries in an attack
    """

    __name__ = "NumQueriesMetric"

    def __init__(self, language: Union[str, LANGUAGE]) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self.all_metrics = {}

    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        self.results = results
        self.num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        self.all_metrics["avg_num_queries"] = self.__avg_num_queries()

        return self.all_metrics

    def __avg_num_queries(self) -> float:
        """ """
        avg_num_queries = self.num_queries.mean()
        avg_num_queries = round(avg_num_queries, 2)
        return avg_num_queries
