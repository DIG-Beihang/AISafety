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

import math
from tqdm import tqdm

class EBD(Evaluation):

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
        super(EBD, self).__init__(outputs_origin, outputs_adv,device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        self.direction_nums = 1000

    def _orthogonal_vectors(self, input_num, vector_num):
        '''

        :param input_num
        :param vector_num
        '''
        randmat = np.random.normal(size=(input_num, vector_num))
        q, _ = np.linalg.qr(randmat)
        vectors = q.T * np.sqrt(float(input_num))

        return vectors
    
    def _mags(self):
        '''

        :param None
        '''
        mags = np.arange(0, 127.5, 2, dtype=np.float32)
        mags = mags / 255.0
        init_mags = mags.copy()
        return mags, init_mags
    
    def _measure_one_image(self, x, y, directions, mags, init_mags, direction_rst_list):
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
        bd_list = []
        C, H, W = x.shape[1:]
        # 对每张图片，计算所有的方向中，最小的BD值
        for index in range(len(directions)):
            direct = directions[index].view(1, C, H, W)
            new_x = var_x + mags * direct
            new_x = torch.clamp(new_x, 0, 1)
            with torch.no_grad():
                output = self.model(new_x)
            pred = torch.argmax(output, 1)
            # 如果全部的扰动都没有造成预测错误，则扰动取最大值
            # 若扰动产生预测错误，搜索最小预测错误扰动值
            epsilon = init_mags[-1]
            if((pred != var_y).data.sum().item() != 0):
                ind = (pred != var_y).cpu().numpy().tolist().index(1)
                epsilon = init_mags[ind]
            
            # 使用2-范数计算，sqrt(每个像素点扰动值^2 / 总像素点数)
            # torch.norm(A) = sqrt(a11^2 + a12^2 + ... + ann^2)
            bd = torch.norm(epsilon * direct / math.sqrt(C * H * W)).cpu().numpy()
            bd_list.append(bd)
            
            # 以方向做ranking时，取该方向所有距离取平均
            direction_rst_list[index] += bd

        # 以图片做ranking时，取所有方向上最小的BD值
        return min(bd_list)

    def evaluate(self,adv_xs=None, cln_xs=None, cln_ys=None,adv_ys=None,target_preds=None, target_flag=False):
        '''
        @description:empirical boundary distance
        @param {
            adv_xs: 攻击样本
            cln_xs：原始样本
            cln_ys: 原始类别，非目标攻击下原始样本的类型
            adv_ys: 攻击样本的预测类别
            target_preds： 目标攻击下希望原始样本攻击的目标类别
            target_flag：是否是目标攻击
        }
        @return: ebd {empirical boundary distance}
        '''
        total = len(adv_xs)
        print("total", total)
        device = self.device
        self.model.eval().to(device)
        N, C, H, W = adv_xs.shape
        vectors = self._orthogonal_vectors(C * H * W, self.direction_nums)
        directions = torch.from_numpy(vectors).float().to(device)
        mags, init_mags = self._mags()
        mags = torch.from_numpy(mags).float().to(device).view(-1, 1, 1, 1)
        
        distance = 0
        image_rst_list = []
        direction_rst_list = [0 for i in range(len(directions))]
        for i in tqdm(range(N)):
            rst = self._measure_one_image(adv_xs[i:i+1], cln_ys[i:i+1], directions, mags, init_mags, direction_rst_list)
            distance += rst
            image_rst_list.append(rst)
        
        # 数据处理部分，分段处理取min(20, k%)的数据做展示
        rate = 0.1
        dir_step = int(len(directions) // min(20, len(directions) * rate))
        img_step = int(total // min(20, total * rate))
        direction_rst_list = [float('{:.4f}'.format(x / total)) for x in direction_rst_list][::dir_step]
        image_rst_list = [float('{:.4f}'.format(x)) for x in image_rst_list][::img_step]
        
        with open("ebd.log", "a") as fp:
            fp.write("-------------------------------\n按Image排序的EBD值\n")
            image_rst_list.sort()
            fp.write("{}\n".format(image_rst_list))
            fp.write("按Direction排序的EBD值\n")
            direction_rst_list.sort()
            fp.write("{}\n".format(direction_rst_list))

        if not N==0:
            ebd = distance / N
        else:
            ebd = distance / (N+MIN_COMPENSATION)
        return ebd