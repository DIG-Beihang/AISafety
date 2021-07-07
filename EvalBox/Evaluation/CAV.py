#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Linna
@LastEditors: Linna
@Description:
@Date: 2019-04-23
@LastEditTime:
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from EvalBox.Evaluation.evaluation_defense import Evaluation_Defense
from EvalBox.Evaluation.acc import ACC
#import pyguetzli
from PIL import Image
import zlib
import os,sys

class CAV(Evaluation_Defense):
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
        super(CAV, self).__init__(outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device)
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
        @return: CAV {accuracy rate}
        '''
        total = len(adv_xs)
        acac_eval_origin, eva_name_origin = ACC(self.outputs_origin,self.outputs_adv,self.device), 'ACC'
        rst_Origin= acac_eval_origin.evaluate(adv_xs, cln_xs, cln_ys,adv_ys,target_preds, target_flag)
        acac_eval_Defense, eva_name_Defense = ACC(self.defense_outputs_origin,self.defense_outputs_adv,self.device), 'ACC'
        rst_Defense = acac_eval_Defense.evaluate(adv_xs, cln_xs, cln_ys,adv_ys,target_preds, target_flag)
        acc=abs(rst_Defense-rst_Origin)
        return acc
