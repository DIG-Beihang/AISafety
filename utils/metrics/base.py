# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2022-03-19

metric 抽象基类，
主要基于TextAttack以及OpenAttack相应的模块进行实现
"""

from abc import ABC, abstractmethod
from typing import NoReturn, Any, Sequence, Union

from ..goal_functions import GoalFunctionResult
from ..strings import LANGUAGE, ReprMixin


__all__ = [
    "Metric",
]


class Metric(ReprMixin, ABC):
    """A metric for evaluating Adversarial Attack candidates."""

    @abstractmethod
    def __init__(self, language: Union[str, LANGUAGE], **kwargs: Any) -> NoReturn:
        """ """
        raise NotImplementedError

    @abstractmethod
    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        """Abstract function for computing any values which are to be calculated as a whole during initialization"""
        raise NotImplementedError
