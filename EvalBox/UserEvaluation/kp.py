import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
from EvalBox.Evaluation.evaluation import Evaluation
from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

class KP(Evaluation):
    def __init__(self, outputs_origin,outputs_adv, device, model, **kwargs):
        '''
        @description:
            该方法计算的是模型关键路径。
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
        super(KP, self).__init__(outputs_origin,outputs_adv, device)
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
        self.features_out_hook = []
        self.key_paths = {}
        self.unit_pick_percent = 30

    def for_hook(self, module, fea_in, fea_out):
        # self.module_name.append(module.__class__)
        # self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)

    def output_feature_map(self):
        # print("*"*5+"hook record features"+"*"*5)
        for i in range(len(self.module_name)):
            print(self.module_name[i])
            print(self.features_out_hook[i].size())
        print("*"*5+"hook record features"+"*"*5)

    def getPath(self, model, loss, N, key_path, use_gpu = False):
        # 把模型的梯度置0
        model.zero_grad()
        # print("===== 3 =====")
        # 选取几个单元
        unit_pick_num = int(self.features_out_hook[N - 1].size(1) * self.unit_pick_percent / 100)
        # print("===== 4 =====")

        weight = torch.ones(loss.size())
        # print("===== 5 =====")

        # for index in range(N):
        #     self.features_out_hook[index].requires_grad_(True)
        # loss.requires_grad_(True)
        if use_gpu:
            loss = loss.cuda()
            weight = weight.cuda()
            for index in range(N):
                self.features_out_hook[index] = self.features_out_hook[index].cuda()
        # 计算由最后一层到loss输出梯度下降结果
        gradient = torch.autograd.grad(outputs = loss,
                                       inputs = self.features_out_hook[N - 1],
                                       grad_outputs = weight,
                                       retain_graph = True,
                                       create_graph = True,
                                       only_inputs = True)
        # print("===== 6 =====")

        # sum部分不太懂====== 得到了梯度结算结
        # print(gradient[0].data.size())
        gra = gradient[0].data.cpu().abs()
        # gra = gradient[0].data.cpu().abs().sum((0, 2, 3))
        # print("===== 7 =====")

        # 得到最关键的前k个单元
        _, units = torch.topk(gra, unit_pick_num)
        # print("===== 8 =====")

        key_path[N - 2] = units.numpy().tolist()
        # print("===== 9 =====")

        # 从第12层一路反推到第1层
        for layer_index in range(N - 2, 0, -1):
            key_path[layer_index - 1] = self.compute_gradient(model, layer_index, key_path[layer_index], use_gpu)
        # print("===== 14 =====")
        return key_path

    def compute_gradient(self, model, layer_index, unit_indexs, use_gpu = False):
        # 把梯度置0
        model.zero_grad()
        # print("===== 10 =====")

        unit_pick_num = int(self.features_out_hook[layer_index].size(1) * self.unit_pick_percent / 100)

        if unit_pick_num < 20 and 20 <= self.features_out_hook[layer_index].size(1):
            unit_pick_num = 20

        first_flag = True
        # print("===== 11 =====")
        for unit_index in unit_indexs:
            # print(self.features_out_hook[layer_index + 1].size())
            loss = self.features_out_hook[layer_index + 1].abs().sum()
            weight = torch.ones(loss.size())

            if use_gpu:
                weight = weight.cuda()
                loss = loss.cuda()
            gradient = torch.autograd.grad(outputs = loss,
                                           inputs = self.features_out_hook[layer_index],
                                           grad_outputs = weight,
                                           retain_graph = True,
                                           create_graph = True,
                                           only_inputs = True)
            if first_flag:
                # gra = gradient[0].data.cpu().abs()
                gra = gradient[0].data.cpu().abs().sum((0, 2, 3))
                first_flag = False
            else:
                # gra += gradient[0].data.cpu().abs()
                gra += gradient[0].data.cpu().abs().sum((0, 2, 3))
        # print("===== 12 =====")
        #     print(gradient[0].data.cpu().abs().sum((0,2,3)))
        # print(gra.size())
        _, index = torch.topk(gra, unit_pick_num)
        # print("===== 13 =====")
        return index.numpy().tolist()

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
        # print("total",total)
        device = self.device
        use_gpu = torch.cuda.is_available()
        self.model = self.model.eval().to(device)
        if use_gpu:
            self.model = self.model.cuda()
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'

        adv_label=[]
        number = 0

        for name, module in self.model._modules.items():
            self.module_name.append(name)

        cln_dataset = Data.TensorDataset(adv_xs, cln_ys)
        cln_loader = Data.DataLoader(cln_dataset, batch_size=self.batch_size, num_workers=3)

        loss_func = nn.CrossEntropyLoss()
        for x, y in cln_loader:
            x, y = Variable(x.to(device)), Variable(y.to(device))
            # with torch.no_grad():
            #     # 获取预测结果，同时self.features_out_hook已存储中间层输出信息
            #     pred = self.model(x)
            for index in range(y.size()[0]):
                test_x = x[index].unsqueeze(0)
                test_y = y[index].unsqueeze(0)
                test_y = torch.tensor(test_y, dtype = torch.long)
                if use_gpu:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                    
                # 监听所有中间层特征
                hook_handle_list = []
                self.features_out_hook.clear()
                for name, module in self.model._modules.items():
                    handle = module.register_forward_hook(self.for_hook)
                    hook_handle_list.append(handle)

                # 执行预测过程
                test_output = self.model(test_x)

                # 释放hook
                for handle in hook_handle_list:
                    handle.remove()

                # 获得当次预测值
                cla = torch.max(test_output.squeeze().data, 0)[1].data.item()

                # print(cla)
                # print(test_output.size())
                # print(test_y.size())
                if use_gpu:
                    loss = loss_func(test_output, test_y).cuda()
                else:
                    loss = loss_func(test_output, test_y)

                # print("===== 1 =====")
                feature_num = len(self.module_name)
                key_path = [None] * feature_num
                key_path[feature_num - 1] = cla
                # print("===== 2 =====")
                key_path = self.getPath(self.model, loss, feature_num, key_path, use_gpu)
                # print("===== 15 =====")
                # print(cla)
                # print(test_y)
                # print(key_path)
                if cla != test_y.item():
                    self.key_paths[index] = key_path
                # print("===== 16 =====")
        print(self.key_paths)
        return 1.0