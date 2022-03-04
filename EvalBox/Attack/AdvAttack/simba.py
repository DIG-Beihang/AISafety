#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Li Haoyuan
@LastEditors: Li Haoyuan
@Description: 
@Date: 2021-07-01 17:00:00
@LastEditTime: 2021-07-02 17:34:00
"""
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack


class SIMBA(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Simple Black-box Adversarial Attacks
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(SIMBA, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            max_iters:
            decay_factor:
        } 
        @return: None
        """
        self.epsilon = float(kwargs.get("epsilon", 0.1))
        self.max_iters = int(kwargs.get("max_iters", 1000))

    def get_pred(self, xs, ys):
        var_xs = Variable(
                torch.from_numpy(xs).float().to(self.device), requires_grad=False
            )
        var_ys = ys.to(self.device)
        logits = torch.nn.functional.softmax(self.model(var_xs), dim = 1)
        pred = torch.argmax(logits,dim=1)
        prob_ys = torch.diag(torch.index_select(logits, 1, var_ys))
        return pred.cpu(), prob_ys.cpu()

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        adv_xs = np.copy(xs.numpy())
        targeted = self.IsTargeted
        batch_size =adv_xs.shape[0]
        input_dim=xs.view((batch_size,-1)).shape[1]
        if self.max_iters>input_dim:
            print("Warning: max_iter larger than input dimension")
            self.max_iters=input_dim
        rand_indices = torch.randperm(input_dim)[:self.max_iters]
        for k in range(self.max_iters):
            pred, prob_ys = self.get_pred(adv_xs, ys)
            dim = rand_indices[k]
            if targeted:
                remaining=ys.ne(pred)
            else:
                remaining=ys.eq(pred)
            if torch.sum(remaining)==0:
                break
            perturbation = np.zeros((batch_size, input_dim))
            perturbation[:, dim]=self.epsilon
            perturbation = perturbation.reshape(adv_xs.shape)
            left_step=adv_xs-perturbation
            right_step=adv_xs+perturbation
            pred_l, prob_l = self.get_pred(left_step, ys)
            pred_r, prob_r = self.get_pred(right_step, ys)
            if targeted:
                lsec=torch.logical_and(remaining, torch.logical_or(torch.eq(pred_l, ys), torch.gt(prob_l, prob_ys)))
                rsec=torch.logical_and(torch.logical_not(lsec), torch.logical_and(remaining, torch.logical_or(torch.eq(pred_r, ys), torch.gt(prob_r, prob_ys))))
            else:
                lsec=torch.logical_and(remaining, torch.logical_or(torch.ne(pred_l, ys), torch.lt(prob_l, prob_ys)))
                rsec=torch.logical_and(torch.logical_not(lsec), torch.logical_and(remaining, torch.logical_or(torch.ne(pred_r, ys), torch.lt(prob_r, prob_ys))))
            adv_xs[lsec]=left_step[lsec]
            adv_xs[rsec]=right_step[rsec]
        return torch.from_numpy(adv_xs)
