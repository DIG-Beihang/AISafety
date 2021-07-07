import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
import cv2
import os
import PIL.Image as Image

from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class SNS(Evaluation):
    def __init__(self, outputs_origin,outputs_adv, device, model, **kwargs):
        '''
        @description:
            该方法计算的是敏感单元。
            在参数设置时应注意 
                1. IS_WHITE为True，白盒攻击
                2. IS_PYTHORCH_WHITE为True
                3. IS_COMPARE_MODEL为False
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        super(SNS, self).__init__(outputs_origin,outputs_adv, device)
        self.model = model
        self.device = device
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
        self.feature_dict = dict()
        self.module_name = []
        # self.features_in_hook = []
        self.features_out_hook = []
        self.clean_feature_map = []

    def for_hook(self, module, fea_in, fea_out):
        # self.module_name.append(module.__class__)
        # self.features_in_hook.append(fea_in)
        self.features_out_hook.append(np.squeeze(fea_out.data.cpu().numpy()))

    def output_feature_map(self):
        print("*"*5+"hook record features"+"*"*5)
        for i in range(len(self.module_name)):
            print(self.module_name[i])
            print(self.clean_feature_map[i].shape)
            print(self.features_out_hook[i].shape)
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
        @return: sns {}
        '''
        #print("target_flag",target_flag)
        total = len(adv_xs)
        print("total",total)
        device = self.device
        self.model = self.model.eval().to(device)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'

        adv_label=[]
        number = 0

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

        # 原始样本获取feature_map
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

        average_diff_l1_dict = {}
        average_diff_l2_dict = {}
        average_diff_ratio_l1_dict = {}
        average_diff_ratio_l2_dict = {}
        useless_neuron = {}
        use_gpu = torch.cuda.is_available()
        # key表示特征名
        for key in range(len(self.module_name)):
            # k 表示第k个测试样本
            for k in range(len(self.clean_feature_map[key])):
                # c表示第c个神经元
                for c in range(len(self.clean_feature_map[key][k])):
                    #load feature maps and compute the corresponding criteria
                    x = self.clean_feature_map[key][k][c].flatten()
                    x_adv = self.features_out_hook[key][k][c].flatten()
                    x_l1norm = np.linalg.norm(x, ord=1) / len(x)
                    x_l2norm = np.linalg.norm(x, ord=2) / len(x)
                    diff_feat = x_adv - x
                    diff_feat_l1norm = np.linalg.norm(diff_feat, ord=1) / len(x)
                    diff_feat_l2norm = np.linalg.norm(diff_feat, ord=2) / len(x)
                    if(x_l1norm != 0):
                        diff_ratio_l1 = diff_feat_l1norm / x_l1norm
                        diff_ratio_l2 = diff_feat_l2norm / x_l2norm
                    else:
                        if(key not in useless_neuron):
                            useless_neuron[key] = [c]
                        else:
                            useless_neuron[key] = list(set(useless_neuron[key]))
                            useless_neuron[key].append(c)
                        diff_ratio_l1 = 1.0
                        diff_ratio_l2 = 1.0

                    # update dicts
                    if key not in average_diff_l1_dict.keys():
                        average_diff_l1_dict[key] = np.zeros(len(self.clean_feature_map[key][k]))
                        average_diff_l2_dict[key] = np.zeros(len(self.clean_feature_map[key][k]))
                        average_diff_ratio_l1_dict[key] = np.zeros(len(self.clean_feature_map[key][k]))
                        average_diff_ratio_l2_dict[key] = np.zeros(len(self.clean_feature_map[key][k]))
                    average_diff_l1_dict[key][c] += diff_feat_l1norm
                    average_diff_l2_dict[key][c]  += diff_feat_l2norm
                    average_diff_ratio_l1_dict[key][c]  += diff_ratio_l1
                    average_diff_ratio_l2_dict[key][c]  += diff_ratio_l2
        
        for key in average_diff_l1_dict:
            average_diff_l1_dict[key] /= total
            average_diff_l2_dict[key] /= total
            average_diff_ratio_l1_dict[key] /= total
            average_diff_ratio_l2_dict[key] /= total
            
        with open("neuron_sensitivity.log", "w") as fp:
            fp.write("-------------------------------\n每层前top 10%神经元敏感度\n")
            print("每层前top 10% (最多展示10个) 神经元敏感度")
            for key in average_diff_l1_dict:
                knum = min(10, max(1, int(len(average_diff_l1_dict[key]) / 10)))
                average_diff_l1_dict[key].sort()
                topk = average_diff_l1_dict[key][-knum:]
                topk = [float('{:.4f}'.format(i)) for i in topk]
                # print(self.module_name[key], topk)
                fp.write("{} {}\n".format(self.module_name[key], topk))

            print("每层mean神经元敏感度")
            fp.write("每层mean神经元敏感度\n")
            for key in average_diff_l1_dict:
                mSNS = format(average_diff_l1_dict[key].mean(), '.4f')
                # print(self.module_name[key], mSNS)
                fp.write("{} {}\n".format(self.module_name[key], mSNS))
            
        # 获取除Conv 及 输出层外，敏感度最高层
        layer_name = -1
        for key in average_diff_l1_dict:
            if "conv" in self.module_name[key].lower() or key == len(average_diff_l1_dict) - 1:
                continue
            if layer_name == -1 or average_diff_l1_dict[layer_name].mean() < average_diff_l1_dict[key].mean():
                layer_name = key
        print(self.module_name[layer_name])
        return 1.0