#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-03 13:39:26
@LastEditTime: 2019-04-08 09:08:50
'''
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

import torch.nn.functional as F

class WBD2(Evaluation):

    def __init__(self, outputs_origin, outputs_adv, device, model=None, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        self.model = model
        super(WBD2, self).__init__(outputs_origin, outputs_adv,device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        pass

    def _mags(self):
        '''

        :param None
        '''
        mags = np.arange(0, 127.5, 2, dtype=np.float32)
        init_mags = mags.copy()
        mags = mags / 255.0
        return mags, init_mags
    
    def _measure_one_image(self, x, y, directions, mags, init_mags):
        '''
        @description: 
        @param {
            x:
            y:
            directions:
            mags:
            init_mags:
        } 
        @return: farthest {the worst boundary distance 2}
        '''
        device = self.device
        var_x = Variable(x.to(device))
        var_y = Variable(y.to(device))
        margin_list = []
        C, H, W = x.shape[1:]

        for dir in directions:
            dir = dir.view(1, C, H, W)
            new_x = var_x + mags * dir.to(device)
            new_x = torch.clamp(new_x, 0, 1)
            with torch.no_grad():
                output = self.model(new_x)
            pred = torch.argmax(output, 1)
            if((pred != var_y).data.sum().item() == 0):
                margin_list.append(255)
            else:
                ind = torch.argmax(((pred != var_y) + 0), 0)
                margin_list.append(int(init_mags[ind]))
        return max(margin_list)

    def evaluate(self,adv_xs=None, cln_xs=None, cln_ys=None,adv_ys=None,target_preds=None, target_flag=False):
        '''
        @description:the worst boundary distance
        @param {
            adv_xs: 攻击样本
            cln_xs：原始样本
            cln_ys: 原始类别，非目标攻击下原始样本的类型
            adv_ys: 攻击样本的预测类别
            target_preds： 目标攻击下希望原始样本攻击的目标类别
            target_flag：是否是目标攻击
        }
        @return: wbd2 {the worst boundary distance 2}
        '''
        total = len(adv_xs)
        print("total", total)

        N, C, H, W = adv_xs.shape
        mags, init_mags = self._mags()
        directions = (adv_xs - cln_xs).view(adv_xs.size(0), -1)
        directions = F.normalize(directions, p=2, dim=1)
        mags = torch.from_numpy(mags).float().to(self.device).view(-1, 1, 1, 1)

        distance = 0
        for i in range(N):
            distance += self._measure_one_image(adv_xs[i:i+1], cln_ys[i:i+1], directions, mags, init_mags)
        if not N == 0:
            wbd = distance / N
        else:
            wbd = distance / (N+MIN_COMPENSATION)
        return wbd