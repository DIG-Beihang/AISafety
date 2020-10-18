#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Zhao Lijun
@LastEditors: Zhao Lijun
@Description:
@Date: 2019-04-22
@LastEditTime: 2019-04-22 14:50
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class PSD(Evaluation):
    def __init__(self, outputs_origin, outputs_adv,device, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(PSD, self).__init__(outputs_origin, outputs_adv,device)

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
        @return: psd {Perturbation Sensitivity Distance}
        '''
        total = len(adv_xs)
        print("total", total)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        dist = 0
        number = 0
        outputs=torch.from_numpy(self.outputs_adv)
        predicts = list()

        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        predicts.extend(preds)
        labels = target_preds.numpy()
        print(target_flag)
        if not target_flag:
            for i in range(len(predicts)):
                if predicts[i] != labels[i]:
                    number += 1
                    image = cln_xs.numpy()[i]
                    pert = abs(adv_xs.numpy()[i] - cln_xs.numpy()[i])

                    for idx_channel in range(image.shape[0]):
                        image_channel = image[idx_channel]
                        pert_channel = pert[idx_channel]

                        image_channel = np.pad(image_channel, 1, 'reflect')
                        pert_channel = np.pad(pert_channel, 1, 'reflect')

                        for i in range(1, image_channel.shape[0] - 1):
                            for j in range(1, image_channel.shape[1] - 1):
                                sd_value=float(np.std(np.array(
                                    [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1],
                                     image_channel[i, j - 1],image_channel[i, j], image_channel[i, j + 1],
                                     image_channel[i + 1, j - 1],image_channel[i + 1, j],image_channel[i + 1, j + 1]])))
                                dist += pert_channel[i, j] *sd_value
        else:
            for i in range(len(predicts)):
                if predicts[i] == labels[i]:
                    number += 1
                    image = cln_xs.numpy()[i]
                    pert = abs(adv_xs.numpy()[i] - cln_xs.numpy()[i])

                    for idx_channel in range(image.shape[0]):
                        image_channel = image[idx_channel]
                        pert_channel = pert[idx_channel]

                        image_channel = np.pad(image_channel, 1, 'reflect')
                        pert_channel = np.pad(pert_channel, 1, 'reflect')

                        for i in range(1, image_channel.shape[0] - 1):
                            for j in range(1, image_channel.shape[1] - 1):
                                sd_value=float(np.std(np.array(
                                    [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1],
                                     image_channel[i, j - 1],image_channel[i, j], image_channel[i, j + 1],
                                     image_channel[i + 1, j - 1],image_channel[i + 1, j],image_channel[i + 1, j + 1]])))
                                dist += pert_channel[i, j] *sd_value
        if not number==0:
            psd = dist/number
        else:
            psd = dist/(number+MIN_COMPENSATION)
        return psd
