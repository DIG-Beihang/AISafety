#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 15:42:20
@LastEditTime: 2019-04-15 09:24:24
"""
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack


class UMIFGSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Untargeted Momentum Iterative Method
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(UMIFGSM, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
            decay_factor:
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.1))
        self.eps_iter = float(kwargs.get("eps_iter", 0.01))
        self.num_steps = int(kwargs.get("num_steps", 15))
        self.decay_factor = float(kwargs.get("decay_factor", 1.0))

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
        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        momentum = 0
        targeted = self.IsTargeted

        for _ in range(self.num_steps):
            var_xs = Variable(
                torch.from_numpy(copy_xs).float().to(device), requires_grad=True
            )
            var_ys = Variable(ys.to(device))

            outputs = self.model(var_xs)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            else:

                loss = self.criterion(outputs, var_ys)
            loss.backward()

            grad = var_xs.grad.data.cpu().numpy()

            momentum = self.decay_factor * momentum + grad

            copy_xs = copy_xs + self.eps_iter * np.sign(momentum)
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs
