# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-23
@LastEditTime: 2022-01-05
"""

from .base import GoalFunction, GoalFunctionResultStatus, GoalFunctionResult
from .classification import (
    ClassificationGoalFunction,
    ClassificationGoalFunctionResult,
    UntargetedClassification,
    TargetedClassification,
    InputReduction,
)


__all__ = [
    "GoalFunction",
    "GoalFunctionResultStatus",
    "GoalFunctionResult",
    "ClassificationGoalFunction",
    "ClassificationGoalFunctionResult",
    "UntargetedClassification",
    "TargetedClassification",
    "InputReduction",
]
