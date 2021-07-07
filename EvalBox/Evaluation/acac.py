#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Zhao Lijun
@LastEditors: Zhao Lijun
@Description:
@Date: 2019-04-19
@LastEditTime: 2019-04-19 16:05
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class ACAC(Evaluation):
    def __init__(self, outputs_origin,outputs_adv, device, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(ACAC, self).__init__(outputs_origin,outputs_adv, device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description:
        @param {
        }
        @return:
        '''

    def evaluate(self,adv_xs=None, cln_xs=None, cln_ys=None,adv_ys=None,target_preds=None, target_flag=False):
        '''
        @description:
        @param {
            adv_xs: 攻击样本
            cln_xs：原始样本
            cln_ys: 原始类别，非目标攻击下原始样本的类型
            adv_ys: 攻击样本的预测类别
            target_preds： 目标攻击下希望原始样本攻击的目标类别
            target_flag：是否是目标攻击
        }
        @return: acac {Average Confidence of Adversarial Class}
        '''
        total = len(adv_xs)
        print("total",total)
        outputs=torch.from_numpy(self.outputs_adv)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        number = 0
        prob = 0
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        outputs_softmax = outputs_softmax.data.numpy()
        preds = preds.data.numpy()
        labels = target_preds.numpy()
        if not target_flag:
            for i in range(preds.size):
                if preds[i] != labels[i]:
                    number += 1
                    prob += np.max(outputs_softmax[i])
        else:
            for i in range(preds.size):
                if preds[i] == labels[i]:
                    number += 1
                    prob += np.max(outputs_softmax[i])
        if not number==0:
            acac = prob/number
        else:
            acac = prob / (number+MIN_COMPENSATION)
        return acac
