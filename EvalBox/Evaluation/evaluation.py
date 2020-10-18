#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-03 14:13:28
@LastEditTime: 2019-04-03 16:19:17
'''
import os
from abc import ABCMeta, abstractmethod
import torch.utils.data as Data
from utils.file_utils import get_user_model
# 防止除数的分母为0
MIN_COMPENSATION=1

class Evaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, outputs_origin,outputs_adv, device):
        '''
        @description: 
        @param {
            model:模型
            device: 设备(GPU)
        } 
        @return: 
        '''
        self.outputs_origin = outputs_origin
        self.outputs_adv = outputs_adv
        self.device = device
        self.batch_size=64

    def prepare_data(self,adv_xs=None, cln_ys=None, target_preds=None, target_flag=False):

        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        if not target_flag:
            dataset = Data.TensorDataset(adv_xs, cln_ys)
        else:
            dataset = Data.TensorDataset(adv_xs, target_preds)
        self.batch_size=dataset.__len__()
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, num_workers=0)

        return  data_loader
    @abstractmethod
    def evaluate(self):
        '''
        @description: abstract method for Evaluations is not implemented
        @param {type} 
        @return: 
        '''
        raise NotImplementedError
