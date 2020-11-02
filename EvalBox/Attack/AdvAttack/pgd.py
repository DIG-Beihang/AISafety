#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 13:46:01
@LastEditTime: 2019-04-15 09:23:44
"""
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack


class PGD(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Projected Gradient Descent (PGD)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(PGD, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.1))
        self.eps_iter = float(kwargs.get("eps_iter", 0.01))
        self.num_steps = int(kwargs.get("num_steps", 15))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted

        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        copy_xs = copy_xs + np.float32(
            np.random.uniform(-self.eps, self.eps, copy_xs.shape)
        )

        for _ in range(self.num_steps):
            var_xs = Variable(
                torch.from_numpy(copy_xs).float().to(device), requires_grad=True
            )
            var_ys = Variable(ys.to(device))

            outputs = self.model(var_xs)
            loss = self.criterion(outputs, var_ys)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            loss.backward()

            grad_sign = var_xs.grad.data.sign().cpu().numpy()
            copy_xs = copy_xs + self.eps_iter * grad_sign
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs
