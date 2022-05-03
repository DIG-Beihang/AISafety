# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-21
@LastEditTime: 2022-03-19

对抗攻击过程中搜索对抗样本方法的抽象基类，
主要基于TextAttack的实现
"""

from abc import ABC, abstractmethod
from typing import Any, List

from ..goal_functions import GoalFunctionResult
from ..transformations import Transformation
from ..strings import ReprMixin
from ...Models.base import NLPVictimModel


__all__ = [
    "SearchMethod",
]


class SearchMethod(ReprMixin, ABC):
    """
    注意，SearchMethod 不能单独使用，需要利用 Transformation, Model 等
    对其添加 get_transformations，get_goal_results 等方法才能使用
    """

    __name__ = "SearchMethod"

    def __call__(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        """Ensures access to necessary functions, then calls ``perform_search``"""
        if not hasattr(self, "get_transformations"):
            raise AttributeError(
                "Search Method must have access to get_transformations method"
            )
        if not hasattr(self, "get_goal_results"):
            raise AttributeError(
                "Search Method must have access to get_goal_results method"
            )
        if not hasattr(self, "filter_transformations"):
            raise AttributeError(
                "Search Method must have access to filter_transformations method"
            )
        # the above attributes are set when initializing an attacker

        result = self.perform_search(initial_result)
        # ensure that the number of queries for this GoalFunctionResult is up-to-date
        result.num_queries = self.goal_function.num_queries
        return result

    @abstractmethod
    def perform_search(self, initial_result: GoalFunctionResult) -> Any:
        """Perturbs `attacked_text` from ``initial_result`` until goal is
        reached or search is exhausted.

        Must be overridden by specific search methods.
        """
        raise NotImplementedError

    def check_transformation_compatibility(
        self, transformation: Transformation
    ) -> bool:
        """Determines whether this search method is compatible with
        ``transformation``."""
        return True

    @property
    def is_black_box(self) -> bool:
        """Returns `True` if search method does not require access to victim
        model's internal states."""
        raise NotImplementedError

    def get_victim_model(self) -> NLPVictimModel:
        if self.is_black_box:
            raise NotImplementedError(
                "Cannot access victim model if search method is a black-box method."
            )
        else:
            return self.goal_function.model

    def extra_repr_keys(self) -> List[str]:
        return []
