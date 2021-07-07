import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class NU(Evaluation):
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
        super(NU, self).__init__(outputs_origin,outputs_adv, device)
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
        self.model_module_dict = dict()
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []
        self.threshold = 0.2

    def for_hook(self, module, fea_in, fea_out):
        # self.module_name.append(module.__class__)
        # self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out.data.cpu().numpy())

    def output_hook(self):
        print("*"*5+"hook record features"+"*"*5)
        for i in range(len(self.module_name)):
            print(self.module_name[i])
            print(self.features_in_hook[i][0].size())
            print(self.features_out_hook[i].size())
        print("*"*5+"hook record features"+"*"*5)

    def init_dict(self):
        for name in self.module_name:
            self.model_module_dict[name] = False

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
        @return: sns {}
        '''
        #print("target_flag",target_flag)
        total = len(adv_xs)
        print("total",total)
        device = self.device

        cln_dataset = Data.TensorDataset(adv_xs, cln_ys)
        cln_loader = Data.DataLoader(cln_dataset, batch_size=self.batch_size, num_workers=3)

        adv_label=[]
        number = 0

        # 监听所有中间层特征
        hook_handle_list = []
        for name, module in self.model._modules.items():
            self.module_name.append(name)
            handle = module.register_forward_hook(self.for_hook)
            hook_handle_list.append(handle)

        # 正常评测
        nc = 0
        for x, y in cln_loader:
            x, y = Variable(x.to(device)), Variable(y.to(device))
            with torch.no_grad():
                outputs = self.model(x)
        
        # 释放hook
        for handle in hook_handle_list:
            handle.remove()

        total = 0
        uncertainty = []
        for key in range(len(self.module_name)):
            # k 表示第k个测试样本
            neuron_output = [[] for i in range(len(self.features_out_hook[key][0]))]
            for k in range(len(self.features_out_hook[key])):
                # c表示第c个神经元
                total += len(self.features_out_hook[key][k])
                for c in range(len(self.features_out_hook[key][k])):
                    neuron_output[c].append(self.features_out_hook[key][k])
            neuron_variance = 0
            for c in range(len(neuron_output)):
                neuron_variance += np.var(neuron_output[c])
            uncertainty.append(float('{:.4f}'.format(neuron_variance / len(neuron_output))))
        for key in range(len(self.module_name)):
            with open("neuron_uncertainty.log", "a") as file:
                file.write(self.module_name[key] + " : " + str(uncertainty[key]) + "\n")
        return 1.0