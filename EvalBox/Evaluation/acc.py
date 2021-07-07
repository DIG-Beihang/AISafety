#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-03 13:38:48
@LastEditTime: 2019-04-09 13:05:08
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class ACC(Evaluation):
    def __init__(self, outputs_origin, outputs_adv, device, **kwargs):
        '''
        @description: 
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        '''
        super(ACC, self).__init__(outputs_origin, outputs_adv, device)

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
            target_preds： 目标攻击下是原始样本攻击的目标类别
            target_flag：是否是目标攻击
        }
        @return: acc {accuracy rate}
        '''
        total = len(adv_xs)
        print("total",total)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        outputs = torch.from_numpy(self.outputs_adv)
        number = 0
        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        labels = target_preds.numpy()
        if target_flag:
            for i in range(preds.size):
                if preds[i] == labels[i]:
                    number += 1
        else:
            for i in range(preds.size):
                if preds[i] != labels[i]:
                    number += 1
        if not total==0:
            acc = number / total
        else:
            acc = number / (total+MIN_COMPENSATION)
        return acc