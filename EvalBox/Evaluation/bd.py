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

class BD(Evaluation):

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
        super(BD, self).__init__(outputs_origin, outputs_adv,device)

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

    def _orthogonal_vectors(self, input_num, vector_num):
        '''

        :param input_num
        :param vector_num
        '''
        randmat = np.random.normal(size=(input_num, vector_num))
        q, _ = np.linalg.qr(randmat)
        vectors = q.T * np.sqrt(float(input_num))

        return vectors
    
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
        @return: farthest {the worst boundary distance}
        '''
        device = self.device
        var_x = Variable(x.float().to(device))
        var_y = Variable(y.to(device,dtype=torch.int64))
        it=0
        margin_list = []
        C, H, W = x.shape[1:]
        for dir in directions:
            dir = dir.view(1, C, H, W)
            new_x = var_x + mags * dir
            new_x = torch.clamp(new_x, 0, 1)
            with torch.no_grad():
                output = self.model(new_x)
            pred = torch.argmax(output, 1)
            if((pred != var_y).data.sum().item() == 0):
                margin_list.append(255)
                it=it+1
            else:

                ind = torch.argmax(((pred != var_y) + 0), 0)
                margin_list.append(int(init_mags[ind]))
                #print(ind,pred, var_y)
        #print(it,margin_list)
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
        @return: wbd {the worst boundary distance}
        '''
        total = len(adv_xs)
        print("total", total)
        device = self.device
        self.model.eval().to(device)
        N, C, H, W = adv_xs.shape
        mags, init_mags = self._mags()
        vectors = self._orthogonal_vectors(C * H * W, 10)
        directions = torch.from_numpy(vectors).float().to(device)
        mags = torch.from_numpy(mags).float().to(device).view(-1, 1, 1, 1)

        distance = 0
        for i in range(N):
            distance += self._measure_one_image(adv_xs[i:i+1], cln_ys[i:i+1], directions, mags, init_mags)
        if not N==0:
            wbd = distance / N
        else:
            wbd = distance / (N+MIN_COMPENSATION)
        return wbd