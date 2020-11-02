#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-26 10:30:19
@LastEditTime: 2019-04-15 11:25:16
"""

import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack

from utils.CrossEntropyLoss import CrossEntropyLoss


class FGSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Fast Gradient Sign Method (FGSM) 
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            IsTargeted:是否是目标攻击
            kwargs: 用户对攻击方法需要的参数
        } 
        @return: None
        """
        super(FGSM, self).__init__(model, device, IsTargeted)
        # 使用该函数时候，要保证训练模型的标签是从0开始，而不是1
        self.criterion = torch.nn.CrossEntropyLoss()
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:沿着梯度方向步长的参数
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.03))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:原始的样本
            ys:样本的标签
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted
        copy_xs = np.copy(xs.numpy())
        var_xs = torch.tensor(
            copy_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_ys = torch.tensor(ys, device=device)

        outputs = self.model(var_xs)
        if targeted:
            loss = -self.criterion(outputs, var_ys)
        else:
            loss = self.criterion(outputs, var_ys)

        loss.backward()
        grad_sign = var_xs.grad.data.sign().cpu().numpy()
        copy_xs = np.clip(copy_xs + self.eps * grad_sign, 0.0, 1.0)
        adv_xs = torch.from_numpy(copy_xs)
        print("FGSM, adv_xs:", adv_xs.device)
        return adv_xs
