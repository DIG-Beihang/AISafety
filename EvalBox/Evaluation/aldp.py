#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Zhao Lijun
@LastEditors: Zhao Lijun
@Description:
@Date: 2019-04-19
@LastEditTime: 2019-04-22 10:38
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class ALDp(Evaluation):
    def __init__(self, outputs_origin, outputs_adv, device, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(ALDp, self).__init__(outputs_origin, outputs_adv, device)

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
        @description:Average L_p Distortion
        @param {
            adv_xs: 攻击样本
            cln_xs：原始样本
            cln_ys: 原始类别，非目标攻击下原始样本的类型
            adv_ys: 攻击样本的预测类别
            target_preds： 目标攻击下是原始样本攻击的目标类别
            target_flag：是否是目标攻击
        }
        @return: L0 distance, norm L2 distance, L_inf distance
        '''
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        ori_r = cln_xs.numpy() * 255
        adv_r = adv_xs.numpy() * 255
        NUM_PIXEL = int(np.prod(cln_xs.shape[1:]))
        pert = adv_r - ori_r
        dist_l0 = 0
        norm_dist_l2 = 0
        dist_li = 0
        number = 0
        predicts = list()
        outputs = torch.from_numpy(self.outputs_adv)
        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        labels = target_preds.numpy()
        predicts.extend(preds)
        if not target_flag:
            for i in range(len(predicts)):
                if predicts[i] != labels[i]:
                    number += 1
                    dist_l0 += (np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL)
                    norm_dist_l2 += np.linalg.norm(np.reshape(cln_xs.numpy()[i] - adv_xs.numpy()[i], -1), ord=2) / \
                               (np.linalg.norm(np.reshape(cln_xs.numpy()[i], -1), ord=2))

                    dist_li += np.linalg.norm(np.reshape(cln_xs.numpy()[i] - adv_xs.numpy()[i], -1), ord=np.inf)
        else:
            for i in range(len(predicts)):
                if predicts[i] == labels[i]:
                    number += 1
                    dist_l0 += (np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL)
                    norm_dist_l2 += np.linalg.norm(np.reshape(cln_xs.numpy()[i] - adv_xs.numpy()[i], -1), ord=2) / \
                                    (np.linalg.norm(np.reshape(cln_xs.numpy()[i], -1), ord=2))
                    dist_li += np.linalg.norm(np.reshape(cln_xs.numpy()[i] - adv_xs.numpy()[i], -1),
                                                  ord=np.inf)
        if not number==0:
            adv_l0 = dist_l0 / number
            norm_adv_l2 = norm_dist_l2 / number
            adv_li = dist_li / number
        else:
            adv_l0 = dist_l0 / (number+MIN_COMPENSATION)
            norm_adv_l2 = norm_dist_l2 / (number+MIN_COMPENSATION)
            adv_li = dist_li / (number+MIN_COMPENSATION)

        return adv_l0, norm_adv_l2, adv_li
