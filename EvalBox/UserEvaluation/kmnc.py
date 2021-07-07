import numpy as np
import torch
import math
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class KMNC(Evaluation):
    def __init__(self, outputs_origin,outputs_adv, device, model, **kwargs):
        '''
        @description:
            该方法计算的是K节神经元覆盖率。
            在参数设置时应注意 
                1. Data_path中前两个，需要放置训练集而非测试集，用于生成神经元阈值范围
                2. IS_WHITE为True，白盒攻击
                3. IS_PYTHORCH_WHITE为True
                4. IS_COMPARE_MODEL为False
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(KMNC, self).__init__(outputs_origin,outputs_adv, device)
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
        self.kmnc_feature_map = []
        self.neuron_deal = []
        self.features_out_hook = []
        self.section_num = 20

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
        @return: kmnc {}
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

        for handle in hook_handle_list:
        	handle.remove()

        # 输出feature map =  feature_num * N * neuron_num * output
        # self.output_feature_map()

        self.lower_feature_map = [[] for i in range(len(self.module_name))]
        self.upper_feature_map = [[] for i in range(len(self.module_name))]
        self.kmnc_feature_map = [[] for i in range(len(self.module_name))]
        self.neuron_deal = [[] for i in range(len(self.module_name))]
        for key in range(len(self.module_name)):
            self.lower_feature_map[key] = [0 for i in range(len(self.clean_feature_map[key][0]))]
            self.upper_feature_map[key] = [0 for i in range(len(self.clean_feature_map[key][0]))]
            self.kmnc_feature_map[key] = [[0 for i in range(self.section_num)] for j in range(len(self.clean_feature_map[key][0]))]
            self.neuron_deal[key] = [[0 for i in range(self.section_num)] for j in range(len(self.clean_feature_map[key][0]))]
        
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

        # 统计测试集测试结果
        for key in range(len(self.module_name)):
#             print(self.module_name[key])
            # k 表示第k个测试样本
            for k in range(len(self.features_out_hook[key])):
                # c 表示第c个神经元
                for c in range(len(self.features_out_hook[key][k])):
                    if self.neuron_deal[key][c] == 1:
                        continue
                    # x 表示key层的神经元c在第k个测试时的输出结果
                    # 对应上下界分别为 self.lower_feature_map[key][c]  self.upper_feature_map[key][c]
#                     x = self.features_out_hook[key][k][c].flatten()
                    section_length = (self.upper_feature_map[key][c] - self.lower_feature_map[key][c]) / self.section_num
                    for section_index in range(self.section_num):
                        if self.kmnc_feature_map[key][c][section_index] != 1:
                            x = self.features_out_hook[key][k][c] - (self.lower_feature_map[key][c] + (section_index + 0.5) * section_length)
                            x = np.abs(x).min()
                            if x <= section_length * 0.5:
                                self.kmnc_feature_map[key][c][section_index] = 1
                    key_c_deal = True
                    for section_index in range(self.section_num):
                        if self.kmnc_feature_map[key][c][section_index] != 1:
                            key_c_deal = False
                            break
                    if key_c_deal:
                        self.neuron_deal[key][c] = 1
        total = 0
        covered = 0
        for key in range(len(self.module_name)):
            # c表示第c个神经元
            for c in range(len(self.clean_feature_map[key][0])):
                # index表示神经元结果的第index段
                for index in range(self.section_num):
                    if self.kmnc_feature_map[key][c][index] == 1:
                        covered += 1
                    total += 1
        print(total, covered)
        return covered / total