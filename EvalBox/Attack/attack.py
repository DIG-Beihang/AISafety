# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-22
@LastEditTime: 2021-09-13

对AttackGenerator进行封装，与其余模块（vision等）保持一致

状态
-----
代码完成        √
手动测试完成    √
自动测试完成    X
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, NoReturn

import torch

from .attack_generator import AttackGenerator
from ...Models.base import NLPVictimModel


__all__ = [
    "Attack",
]


class Attack(AttackGenerator, ABC):
    """ """

    __name__ = "Attack"

    def __init__(
        self,
        model: NLPVictimModel,
        device: Optional[torch.device] = None,
        IsTargeted: bool = False,
        **kwargs: Any
    ) -> NoReturn:
        """
        @description:
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            IsTargeted:是否是目标攻击
            }
        @return: None
        """
        self.model = model
        self.device = device
        self.IsTargeted = IsTargeted
        self.batch_size = 1
        self.init_model(device)
        super().__init__(**kwargs)

    def init_model(self, device: Optional[torch.device] = None):
        """ """
        pass

    def _parse_params(self, **kwargs: Any) -> NoReturn:
        """ """
        pass

    def prepare_data(
        self, adv_xs=None, cln_ys=None, target_preds=None, target_flag=False
    ):
        """ """
        pass

    @abstractmethod
    def generate(self):
        """
        @description: Abstract method
        @param {type}
        @return:
        """
        raise NotImplementedError
