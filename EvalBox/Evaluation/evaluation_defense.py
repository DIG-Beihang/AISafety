#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Linna
@LastEditors: Linna
@Description:
@Date: 2019-04-23
@LastEditTime: 2019-04-24
'''
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from EvalBox.Evaluation.evaluation import Evaluation
from utils.file_utils import get_user_model,get_user_model_origin

class Evaluation_Defense(Evaluation):
    __metaclass__ = ABCMeta

    def __init__(self, outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device):
        '''
        @description:
        @param {
            model:
        }
        @return:
        '''
        self.outputs_origin = outputs_origin
        self.outputs_adv = outputs_adv
        self.defense_outputs_origin = defense_outputs_origin
        self.defense_outputs_adv = defense_outputs_adv
        self.device = device
        self.batch_size=64

    def get_Preds_Result(self,outputs):
        tensor_outputs=torch.from_numpy(outputs)
        outputs_softmax= torch.nn.functional.softmax(tensor_outputs, dim=1)
        preds = torch.argmax(tensor_outputs, 1)
        outputs_softmax = outputs_softmax.data.cpu().numpy()
        preds = preds.data.cpu().numpy()
        return outputs_softmax,preds,tensor_outputs

    def succesfulfilter(self,adv_xs, adv_ys, cln_xs, cln_ys, filter_Flag=True,target_FLag=False):
        adv_data = []
        adv_label = []
        cln_data = []
        cln_label = []
        if filter_Flag:
            number = 0
            targets = adv_ys.data.cpu().numpy()
            labels = cln_ys.data.cpu().numpy()
            adv_xn = adv_xs.data.cpu().numpy()
            cln_xn = cln_xs.data.cpu().numpy()

            for i in range(targets.size):
                if not target_FLag:
                  if targets[i] != labels[i]:
                     adv_xadd = adv_xn[i][np.newaxis, :]
                     cln_xadd = cln_xn[i][np.newaxis, :]
                     adv_data.extend(adv_xadd)
                     adv_label.append(targets[i])
                     cln_data.extend(cln_xadd)
                     cln_label.append(labels[i])
                     number = number + 1

                else:
                  if targets[i] == labels[i]:
                     adv_xadd = adv_xn[i][np.newaxis, :]
                     cln_xadd = cln_xn[i][np.newaxis, :]
                     adv_data.extend(adv_xadd)
                     adv_label.append(targets[i])
                     cln_data.extend(cln_xadd)
                     cln_label.append(labels[i])
                     number = number + 1

            return torch.from_numpy(np.array(adv_data)), torch.from_numpy(np.array(adv_label)), torch.from_numpy(
                np.array(cln_data)), torch.from_numpy(np.array(cln_label))
        else:
            return adv_xs, adv_ys, cln_xs, cln_ys
    def ACCfilter(self,outputs,outputs_defense,
                  cln_ys, target_FLag=False):
        assert len(outputs) == len(cln_ys), 'examples and labels do not match.'
        adv_label = []
        number_defense_success_success = 0
        number_defense_fail_success = 0
        number_defense_fail_fail = 0
        number_defense_success_fail = 0
        outputs_tensor=Variable(torch.from_numpy(outputs).to(self.device))
        preds = torch.argmax(outputs_tensor, 1)
        preds = preds.data.cpu().numpy()
        outputs_defense_tensor=Variable(torch.from_numpy(outputs_defense).to(self.device))
        preds_defense = torch.argmax(outputs_defense_tensor, 1)
        preds_defense = preds_defense.data.cpu().numpy()
        origin_labels = cln_ys.data.cpu().numpy()
        #if not target_FLag:
        for i in range(preds.size):
            adv_label.append(preds[i])
            if preds[i] == origin_labels[i]:
                    #同时分类正确
                if preds_defense[i]== origin_labels[i]:
                    number_defense_success_success += 1
                    #原始正确，防御不成功
                else:
                    number_defense_fail_success+=1
            else:
                # 原始的失败，防御的成功
                if preds_defense[i] == origin_labels[i]:
                    number_defense_success_fail += 1
                # 原始也不成功，防御不成功
                else:
                    number_defense_fail_fail += 1
        # else:
        #     for i in range(preds.size):
        #         adv_label.append(preds[i])
        #         if preds[i] != origin_labels[i]:
        #             # 同时分类正确
        #             if preds_defense[i] != origin_labels[i]:
        #                 number_defense_success_success += 1
        #             # 原始正确，防御不成功
        #             else:
        #                 number_defense_fail_success += 1
        #         else:
        #             # 原始的失败，防御的成功
        #             if preds_defense[i] != origin_labels[i]:
        #                 number_defense_success_fail += 1
        #             # 原始也不成功，防御不成功
        #             else:
        #                 number_defense_fail_fail += 1

        return number_defense_success_success,number_defense_success_fail, number_defense_fail_success,number_defense_fail_fail

    #计算攻击失败的，也就是防御成功的个数
    def DefenseRatefilter(self,outputs,defense_outputs,adv_xs,
                  cln_ys, target_FLag=False):

        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        total = len(adv_xs)
        adv_label = []
        number = 0
        preds = torch.argmax(outputs, 1)
        preds = preds.data.cpu().numpy()
        predsdefense = torch.argmax(defense_outputs, 1)
        predsdefense = predsdefense.data.cpu().numpy()

        origin_labels = cln_ys.data.cpu().numpy()
        adv_xs_npy =  adv_xs.data.cpu().numpy()
        for i in range(preds.size):
              adv_label.append(preds[i])
              #防御方法分类正确
              if predsdefense[i] == origin_labels[i]:
                    #防御成功，但是原始的模型分类不正确
                  if preds[i] != origin_labels[i]:
                    number += 1
        return number

    @abstractmethod
    def evaluate(self):
        '''
        @description: abstract method for Evaluations is not implemented
        @param {type}
        @return:
        '''
        raise NotImplementedError