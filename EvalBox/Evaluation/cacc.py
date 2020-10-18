#!/usr/bin/env python
# coding=UTF-8
'''
@Author: linna
@LastEditors:  linna
@Description: 
@Date: 2020-7-28 09:39:48
@LastEditTime: 2020-7-28 10:38:48
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION
class CACC(Evaluation):
    def __init__(self, outputs_origin, outputs_adv,device, **kwargs):
        '''
        @description: 
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        '''
        super(CACC, self).__init__(outputs_origin, outputs_adv,device)

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
        @return: acc {accuracy rate}
        '''
        total = len(cln_xs)
        print("total",total)
        assert len(cln_xs) == len(cln_ys), 'examples and labels do not match.'

        outputs = torch.from_numpy(self.outputs_origin)
        adv_label=[]
        number = 0

        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        origin_labels = cln_ys.numpy()

        for i in range(preds.size):
            adv_label.append(preds[i])
            if preds[i] == origin_labels[i]:
                 number += 1

        if not total==0:
            cacc = number / total
        else:
            cacc = number / (total+MIN_COMPENSATION)
        return cacc
