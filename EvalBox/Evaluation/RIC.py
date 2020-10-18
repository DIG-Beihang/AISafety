#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Linna
@LastEditors: Linna
@Description:
@Date: 2019-04-23
@LastEditTime:
'''
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from PIL import Image
import zlib
import os,sys

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class RIC(Evaluation):
    def __init__(self, outputs_origin, outputs_adv,device,model=None, **kwargs):
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
        super(RIC, self).__init__(outputs_origin, outputs_adv, device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description:
        @param {
            batch_size:
        }
        @return:
        '''
        self.batch_size = kwargs.get('batch_size', 64)
        self.quality=kwargs.get('quality', 55)

    def _image_compress_transform(self,advSample, quality):
        sample = np.transpose(np.round(advSample * 255), (1, 2, 0))
        image = Image.fromarray(np.uint8(sample))
        saved_adv_image_path=sys.path[0]+"/test0.jpg"
        image.save(saved_adv_image_path,format="JPEG", quality=quality)
        IC_image=Image.open(saved_adv_image_path,"r")
        IC_image = np.transpose(np.array(IC_image), (2, 0, 1)).astype('float32') / 255.0
        os.remove(saved_adv_image_path)
        return IC_image

    def count_numbers(self,var_xs, var_ys,target_flag):
        number = 0
        with torch.no_grad():
            outputs = self.model(var_xs)
            preds = torch.argmax(outputs, 1)
            preds = preds.data.cpu().numpy()
            labels = var_ys.data.cpu().numpy()
            if target_flag:
                for i in range(preds.size):
                    if preds[i] == labels[i]:
                        number += 1
            else:
                for i in range(preds.size):
                    if preds[i] != labels[i]:
                        number += 1


        return number
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
        @return: RIC {accuracy rate}
        '''
        total = len(adv_xs)
        print("total", total)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        device=self.device
        data_loader= self.prepare_data(adv_xs, cln_ys, target_preds, target_flag)




        number = 0
        for xs, ys in data_loader:
            n_xs=torch.Tensor(xs.shape)
            i=0
            for samplex in xs:
                gb_image=self._image_compress_transform(samplex,self.quality)
                torch_xs=torch.from_numpy(gb_image)
                n_xs[i]=torch_xs
                i=i+1
            var_xs, var_ys = Variable(n_xs.to(device)), Variable(ys.to(device))
            numbercount=self.count_numbers(var_xs, var_ys,target_flag)
            number+=numbercount
        if not total==0:
            acc = number / total
        else:
            acc=number / (total+MIN_COMPENSATION)
        return acc
