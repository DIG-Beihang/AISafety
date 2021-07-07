import numpy as np
import torch
import math
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class SNAC(Evaluation):
    def __init__(self, outputs_origin,outputs_adv, device, model, **kwargs):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(SNAC, self).__init__(outputs_origin,outputs_adv, device)
        self.model = model
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
        self.module_name = []
        self.clean_feature_map = []
        self.upper_feature_map = []
        self.lower_feature_map = []
        self.snac_feature_map = []
        self.features_out_hook = []

    def for_hook(self, module, fea_in, fea_out):
        self.features_out_hook.append(fea_out.data.cpu().numpy())

    def output_hook(self):
        print("*"*5+"hook record features"+"*"*5)
        for i in range(len(self.module_name)):
            print(self.module_name[i])
            print(self.features_out_hook[i].size())
        print("*"*5+"hook record features"+"*"*5)

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
        @return: snac {}
        '''
        #print("target_flag",target_flag)
        total = len(adv_xs)
        print("total",total)
        device = self.device
        self.model = self.model.eval().to(device)
        assert len(adv_xs) == len(adv_ys), 'examples and labels do not match.'
        assert len(cln_xs) == len(cln_ys), 'examples and labels do not match.'

        # 监听所有中间层特征
        hook_handle_list = []
        for name, module in self.model._modules.items():
            self.module_name.append(name)
            handle = module.register_forward_hook(self.for_hook)
            hook_handle_list.append(handle)

        cln_dataset = Data.TensorDataset(cln_xs, cln_ys)
        adv_dataset = Data.TensorDataset(adv_xs, adv_ys)
        cln_loader = Data.DataLoader(cln_dataset, batch_size=self.batch_size, num_workers=3)
        adv_loader = Data.DataLoader(adv_dataset, batch_size=self.batch_size, num_workers=3)

        for x, y in cln_loader:
            x, y = Variable(x.to(device)), Variable(y.to(device))
            with torch.no_grad():
                pred = self.model(x)
        self.clean_feature_map = self.features_out_hook.copy()
        self.features_out_hook.clear()

        for x, y in adv_loader:
            x, y = Variable(x.to(device)), Variable(y.to(device))
            with torch.no_grad():
                pred = self.model(x)

        # 释放hook
        for handle in hook_handle_list:
            handle.remove()

        # 输出feature map =  feature_num * N * neuron_num * output
        # self.output_feature_map()

        self.lower_feature_map = [[] for i in range(len(self.module_name))]
        self.upper_feature_map = [[] for i in range(len(self.module_name))]
        self.snac_feature_map = [[] for i in range(len(self.module_name))]
        for key in range(len(self.module_name)):
            self.lower_feature_map[key] = [0 for i in range(len(self.clean_feature_map[key][0]))]
            self.upper_feature_map[key] = [0 for i in range(len(self.clean_feature_map[key][0]))]
            self.snac_feature_map[key] = [0 for i in range(len(self.clean_feature_map[key][0]))]
        
        # 获取训练集产生的[lower, upper]
        # key表示特征名
        for key in range(len(self.module_name)):
            # k 表示第k个测试样本
            for k in range(len(self.clean_feature_map[key])):
                # c表示第c个神经元
                for c in range(len(self.clean_feature_map[key][k])):
                    # x表示key层的神经元c在第k个测试时的输出结果
                    if k == 0:
                        self.lower_feature_map[key][c] = self.clean_feature_map[key][k][c].min()
                        self.upper_feature_map[key][c] = self.clean_feature_map[key][k][c].max()
                    lower_bound = self.clean_feature_map[key][k][c].min()
                    upper_bound = self.clean_feature_map[key][k][c].max()
                    if self.lower_feature_map[key][c] > lower_bound:
                        self.lower_feature_map[key][c] = lower_bound
                    if self.upper_feature_map[key][c] < upper_bound:
                        self.upper_feature_map[key][c] = upper_bound

        # 输出上下界结果
        # for key in range(len(self.module_name)):
        #     print("======", self.module_name[key], "======")
        #     for c in range(len(self.clean_feature_map[key][0])):
        #         print(self.lower_feature_map[key][c], self.upper_feature_map[key][c])

        total = 0
        covered = 0
        # 统计测试集测试结果
        for key in range(len(self.module_name)):
            # k 表示第k个测试样本
            for k in range(len(self.features_out_hook[key])):
                # c表示第c个神经元
                for c in range(len(self.features_out_hook[key][k])):
                    if self.snac_feature_map[key][c] == 1:
                        continue
                    # x表示key层的神经元c在第k个测试时的输出结果
                    # 对应上下界分别为 self.lower_feature_map[key][c]  self.upper_feature_map[key][c]
                    if self.features_out_hook[key][k][c].max() > self.upper_feature_map[key][c]:
                        self.snac_feature_map[key][c] = 1
        total = 0
        covered = 0
        for key in range(len(self.module_name)):
            # c表示第c个神经元
            for c in range(len(self.features_out_hook[key][0])):
                if self.snac_feature_map[key][c] == 1:
                    covered += 1
                total += 2
        return covered / total