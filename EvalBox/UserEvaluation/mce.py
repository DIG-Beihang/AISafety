import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from EvalBox.Evaluation.evaluation_defense import Evaluation_Defense
from EvalBox.Evaluation.acc import ACC
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION
class MCE(Evaluation_Defense):
    def __init__(self, outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device, **kwargs):
        '''
        @description:
            在参数设置时应注意 
                1. IS_COMPARE_MODEL为True
                2. IS_PYTHORCH 为False
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(MCE, self).__init__(outputs_origin,outputs_adv, defense_outputs_origin,defense_outputs_adv,device)

        self._parsing_parameters(**kwargs)

    def _parsing_parameters(self, **kwargs):
        '''
        @description:
        @param {
            batch_size:
        }
        @return:
        '''
        pass

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
        @return: mCE
        '''
        total = len(adv_xs)
        # 作为Base线的模型
        mce_eval_origin = ACC(self.outputs_origin, self.outputs_adv, self.device)
        rst_Origin = 1 - mce_eval_origin.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_preds, target_flag)

        # 用户模型
        mce_eval_Defense = ACC(self.defense_outputs_origin,self.defense_outputs_adv,self.device)
        rst_Defense = 1 - mce_eval_Defense.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_preds, target_flag)
        
        if rst_Origin != 0:
            mce = rst_Defense / rst_Origin
        else:
            mce = rst_Defense / (rst_Origin + MIN_COMPENSATION)
        return mce
