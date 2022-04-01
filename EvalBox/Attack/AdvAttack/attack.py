#!/usr/bin/env python
# coding=UTF-8

from abc import ABCMeta
from abc import abstractmethod
import torch.utils.data as Data
import torch
import torchvision
import importlib

class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, device,IsTargeted):
        '''
        @description: 
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            IsTargeted:是否是目标攻击
            }
        @return: None
        '''
        self.model = model
        self.device = device
        self.IsTargeted=IsTargeted
        self.init_model(device)

    def init_model(self,device):
#         self.model.eval().to(device)
        pass

    def prepare_data(self,adv_xs=None, cln_ys=None, target_preds=None, target_flag=False):
        device = self.device
        self.init_model(device)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        if not target_flag:
            dataset = Data.TensorDataset(adv_xs, cln_ys)
        else:
            dataset = Data.TensorDataset(adv_xs, target_preds)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, num_workers=1)
        return  data_loader,device


    def get_model(self,model_dir,model_name,device):
        #使用预训练的网络，这个网络是ＩｍａｇｅＮｅｔ数据集上面的
        #看看模型是不是默认pytorch的格式
        if model_dir == '':
            model = eval(model_name)(pretrained=True)
        else :
            module_user = importlib.import_module(model_name)
            model = module_user.getModel()
            model.load_state_dict(torch.load(model_dir,map_location=device))
        model = model.eval().to(device)
        return model


    @abstractmethod
    def generate(self):
        '''
        @description: Abstract method
        @param {type} 
        @return: 
        '''
        raise NotImplementedError