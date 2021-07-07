#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Zhao Lijun
@LastEditors: Zhao Lijun
@Description:
@Date: 2019-04-22
@LastEditTime: 2019-04-22 14:02
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from skimage.measure import compare_ssim as SSIM
from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class ASS(Evaluation):
    def __init__(self,outputs_origin, outputs_adv,device, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(ASS, self).__init__(outputs_origin, outputs_adv, device)
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
        @return: ass {Average Structural Similarity}
        '''
        total = len(adv_xs)
        print("total", total)
        ori_r_channel = np.transpose(np.round(cln_xs.numpy() * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        adv_r_channel = np.transpose(np.round(adv_xs.numpy() * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        totalSSIM = 0
        number = 0
        predicts = list()
        outputs = torch.from_numpy(self.outputs_adv)
        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        predicts.extend(preds)
        labels = target_preds.numpy()
        for i in range(len(predicts)):
            if predicts[i] != labels[i]:
                number += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)
                print(SSIM(X=ori_r_channel[i], Y=ori_r_channel[i], multichannel=True))
        if not number==0:
            ass = totalSSIM / number
        else:
            ass = totalSSIM / (number + MIN_COMPENSATION)
        return ass

