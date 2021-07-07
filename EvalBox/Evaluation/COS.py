#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Linna
@LastEditors: Linna
@Description:
@Date: 2019-04-25
@LastEditTime: 2019-04-25
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import scipy.stats as jslib
from PIL import Image
from PIL import ImageFilter
import os
from EvalBox.Evaluation.evaluation_defense import Evaluation_Defense
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION
class COS(Evaluation_Defense):
    def __init__(self, outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device,**kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(COS, self).__init__(outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device)

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
        @return: COS {accuracy rate}
        '''
        total = len(adv_xs)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        print("total", total)
        number = 0
        prob = 0
        prob_defense = 0
        js_total=0
        labels = target_preds.numpy()
        outputs_softmax,preds,outputs=self.get_Preds_Result(self.outputs_adv)
        outputs_softmax_defense, preds_defense,outputs_defense = self.get_Preds_Result(self.defense_outputs_adv)
        for i in range(preds.size):
            #分类正确在原始模型上的
            if not target_flag:
                if preds[i] != labels[i]:
                # 分类正确在防御模型上的
                    if preds_defense[i]!=labels[i]:
                        prob = outputs_softmax[i]
                        prob_defense = outputs_softmax_defense[i]
                        M = (prob+prob_defense) / 2.
                        js = 0.5 * jslib.entropy(prob, M ) + 0.5 * jslib.entropy(prob_defense,M)
                        js_total += js
                        number+=1
            else:
                #目标攻击
                if preds[i] == labels[i]:
                # 分类正确在防御模型上的
                    if preds_defense[i]==labels[i]:
                        prob= outputs_softmax[i]
                        prob_defense = outputs_softmax_defense[i]
                        M = (prob+prob_defense) / 2.
                        js = 0.5 * jslib.entropy(prob, M ) + 0.5 * jslib.entropy(prob_defense,M)
                        js_total += js
                        number+=1
        if not number==0:
            cos=js_total/number
        else:
            cos = js_total / (number+MIN_COMPENSATION)
        return cos