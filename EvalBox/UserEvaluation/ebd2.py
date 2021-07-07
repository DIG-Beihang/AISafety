#!/usr/bin/env python
# coding=UTF-8
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

import torch.nn.functional as F

class EBD2(Evaluation):

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
        super(EBD2, self).__init__(outputs_origin, outputs_adv,device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _measure_one_image(self, xs, ys):
        '''
        @description: 
        @param {
            x:
            y:
        } 
        @return: farthest {empirical boundary distance 2}
        '''
        device = self.device
        copy_xs = np.copy(xs.numpy())
        
        # 限制最大迭代次数
        step_limit = 1000
        # BIM迭代次数
        num_step = 0
        # 步长系数
        eps_iter = 0.0005
        
        # 获取clean数据预测结果
        var_image = Variable(xs.to(device)).float()
        with torch.no_grad():
            outputs = self.model(var_image)
            outputs = outputs.data.cpu().numpy()
            preds_clean = np.argmax(outputs, 1)

        # BIM迭代，直到超限/预测结果变化
        while num_step < step_limit:
            # 检测当前状态下的outputs是否发生变化
            var_xs = torch.tensor(copy_xs, dtype=torch.float, device=device, requires_grad=True)
            var_ys = torch.tensor(ys, device=device)

            outputs = self.model(var_xs)
            preds = torch.argmax(outputs, 1).data.cpu().numpy()
            if preds != preds_clean:
                break
            
            loss = self.criterion(outputs, var_ys)
            loss.backward()

            grad_sign = var_xs.grad.data.sign().cpu().numpy()
            copy_xs = copy_xs + eps_iter * grad_sign
            copy_xs = np.clip(copy_xs, 0.0, 1.0)
            num_step += 1
        return num_step
        

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
        @return: ebd2 {empirical boundary distance 2}
        '''
        total = len(adv_xs)
        print("total", total)

        distance = 0
        out_of_steps = 0
        image_rst_list = []
        for i in range(total):
            rst = self._measure_one_image(adv_xs[i:i+1], cln_ys[i:i+1])
            image_rst_list.append(rst)
            if rst == 1000:
                total -= 1
                out_of_steps += 1
            else:
                distance += rst
        print("总计", total, "个样本中，迭代次数超限", out_of_steps, "次")
            
        if not total == 0:
            ebd = distance / total
        else:
            ebd = distance / (total+MIN_COMPENSATION)
        return ebd