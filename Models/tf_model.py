# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-01
@LastEditTime: 2021-09-01

封装待评测的TensorFlow模型
"""

from .base import NLPVictimModel


__all__ = [
    "TensorFlowNLPVictimModel",
]


class TensorFlowNLPVictimModel(NLPVictimModel):
    """ """

    __name__ = "TensorFlowNLPVictimModel"
