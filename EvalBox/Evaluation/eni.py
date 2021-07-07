#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-03 13:39:15
@LastEditTime: 2019-04-09 12:53:41
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION


class ENI(Evaluation):
    def __init__(self, outputs_origin, outputs_adv,device, **kwargs):
        '''
        @description: 
        @param {
            model:
            device:
        } 
        @return: None
        '''
        super(ENI, self).__init__(outputs_origin, outputs_adv,device)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description: 
        @param {
        } 
        @return: 
        '''

    def evaluate(self,adv_xs=None, cln_xs=None,cln_ys=None,adv_ys=None,target_preds=None,target_flag=False):
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
        @return: eni {emperical noise insensitivity}
        '''
        total = len(adv_xs)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        print("total",total)
        criterion = self.criterion
        total_num = cln_xs.shape[0]
        outputs_adv = torch.from_numpy(self.outputs_adv)
        outputs_origin = torch.from_numpy(self.outputs_origin)
        outputs_adv=Variable(outputs_adv.to(self.device))
        outputs_origin = Variable(outputs_origin.to(self.device))
        cln_ys = cln_ys.to(self.device, dtype=torch.int64)
        #adv_ys=adv_ys.to(self.device, dtype=torch.int64)
        adv_ys = target_preds.to(self.device, dtype=torch.int64)
        loss_cln=criterion(outputs_origin, cln_ys).item()
        loss_adv=criterion(outputs_adv, adv_ys).item()

        dist = torch.max(abs(cln_xs - adv_xs).view(total_num, -1)).sum()

        if not dist==0:
            eni = abs(loss_cln - loss_adv) / dist
        else:
            eni = abs(loss_cln - loss_adv) / (dist+MIN_COMPENSATION)
        return eni.item()/total_num

