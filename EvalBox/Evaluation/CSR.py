#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Linna
@LastEditors: Linna
@Description:
@Date: 2019-04-24
@LastEditTime:
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from EvalBox.Evaluation.acc import ACC
#import pyguetzli
from PIL import Image
import zlib
import os,sys
from EvalBox.Evaluation.evaluation_defense import Evaluation_Defense
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class CSR(Evaluation_Defense):
    def __init__(self, outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(CSR, self).__init__(outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device)
        self._parsing_parameters(**kwargs)
    def _parsing_parameters(self, **kwargs):
        '''
        @description:
        @param {
            batch_size:
        }
        @return:
        '''
    def  evaluate(self,adv_xs=None, cln_xs=None, cln_ys=None,adv_ys=None,target_preds=None, target_flag=False):
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
        @return: CRR,CSR {accuracy rate}
        '''
        total = len(adv_xs)
        print("total", total)
        number_count=0
        number_defense_success_success,number_defense_success_fail,\
        number_defense_fail_success,number_defense_fail_fail=self.ACCfilter(self.outputs_adv,self.defense_outputs_adv,target_preds, target_flag)
        number_count =number_defense_fail_success
        if not total==0:
            acc_CSR=(number_count)/total
        else:
            acc_CSR = (number_count) / (total+MIN_COMPENSATION)

        return acc_CSR
