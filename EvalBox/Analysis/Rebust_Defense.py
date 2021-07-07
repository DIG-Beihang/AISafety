import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision import utils as vutils

sys.path.append("{}/../".format(os.path.dirname(os.path.realpath(__file__))))
from EvalBox.Analysis.evaluation_base import Evaluation_Base
from EvalBox.Defense import *
from torchvision.models import *

from utils.file_utils import (
    get_user_model,
    get_user_model_origin,
    get_user_model_defense,
)
from utils.file_utils import xmlparser
from utils.io_utils import configurate_Device
from Models.UserModel import *

class Rebust_Defense(Evaluation_Base):
    def __init__(self, defense_method = None, sample_path = None, label_path = None, \
                 image_origin_path = None, label_origin_path = None, \
                 gpu_counts = None, gpu_indexs = None, seed = None, \
                 Scale_ImageSize = None, Crop_ImageSize = None, kwargs = None):

        self._parse_params(kwargs)
        super(Rebust_Defense, self).__init__(defense_method, \
            sample_path, label_path, image_origin_path, label_origin_path, \
            gpu_counts, gpu_indexs, seed, Scale_ImageSize, Crop_ImageSize, \
            model = None, model_dir = None, defense_model = None, model_defense_dir = None, \
            data_type = None, IS_WHITE = None, IS_SAVE = None, IS_COMPARE_MODEL = None, \
            IS_TARGETTED = None, save_path = None, save_method = None, \
            black_Result_dir = None, batch_size = None)  # some model that in different setting

    def _parse_params(self, kwargs):
        self.model_name = kwargs.model  # 'resnet20_cifar')
        self.data_type = kwargs.data_type  # 'CIFAR10')
        self.batch_size = kwargs.batch_size  # 64)
        self.Scale_ImageSize = kwargs.Scale_ImageSize
        self.Crop_ImageSize = kwargs.Crop_ImageSize
        self.Enhanced_model_save_path = kwargs.Enhanced_model_save_path
        self.defense_method = kwargs.defense_method
        self.config_defense_param_xml_dir = kwargs.config_defense_param_xml_dir
        self.optim_config_dir = kwargs.optim_config_dir
        self.config_model_dir_path = kwargs.config_model_dir_path
        self.model = None  # 自己生成对抗样本的原模型
        self.device = None

    def test(self, test_loader, model, device):
        model = model.to(device).eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
            val_acc = correct / total
        return val_acc

    def return_model(self):
        return self.model

    def load_optim_config_file(self, model):
        optimizer_string = self.get_parameter(self.optim_config_dir, "optimizer")
        scheduler_string = self.get_parameter(self.optim_config_dir, "scheduler")
        optimizer = eval(optimizer_string)
        scheduler = eval(scheduler_string)
        return optimizer, scheduler

    def gen_dataloader(self):
        # 这里用的是我们自己已经准备的一些已知网络结构的模型
        device, dataloader, dataset = self.setting_data(self.Scale_ImageSize, self.sample_path, self.label_path, self.image_origin_path, self.label_origin_path)
        return device, dataloader, dataset

    def gen_dataloader_train(self):
        # 这里用的是我们自己已经准备的一些已知网络结构的模型
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()
        ])
        dataloader, dataset = self.setting_dataset(self.Scale_ImageSize, self.sample_path, self.label_path, transform)
        return dataset

    def gen_dataloader_test(self):
        # 这里用的是我们自己已经准备的一些已知网络结构的模型
        image_valid_path = self.image_origin_path
        label_valid_path = self.label_origin_path
        dataloader, dataset = self.setting_dataset(self.Scale_ImageSize, image_valid_path, label_valid_path)
        return dataset

    def gen_defense(self, train_loader, valid_loader):
        device, model, dfs, dfs_name = self.load_model_and_config(self.model_name)
        self.model = model
        self.device = device
        # print( self.Enhanced_model_save_path)
        defense_enhanced_saver = (self.Enhanced_model_save_path + "/{}/{}_{}_enhanced.pt".format(dfs_name, self.data_type, dfs_name))
        acc = dfs.generate(train_loader, valid_loader, defense_enhanced_saver)
        print("defensed by {}, result-acc:{}".format(dfs_name, acc))
        return acc, defense_enhanced_saver

    def gen_valid_result(self, valid_loader, defense_enhanced_saver):
        self.config_device()
        model = self.model
        model.load(defense_enhanced_saver, device=self.device)
        acc = self.test(valid_loader, model, self.device)
        return acc

    def get_model_param(self, model_name, device, config_model_dir_path):
        args = xmlparser(config_model_dir_path)
        Model_instance = get_user_model_defense(model_name, **args)
        model = Model_instance.to(device)
        return model

    def config_device(self):
        device = configurate_Device(self.seed, self.gpu_counts, self.gpu_indexs)
        self.device = device
        return self.device

    def load_model(self, model_name):
        self.config_device()
        model = self.get_model_param(model_name, self.device, self.config_model_dir_path)
        self.model = model
        return

    def load_model_and_config(self, model_name):
        device = self.config_device()
        model = self.get_model_param(model_name, device, self.config_model_dir_path)
        defense = None
        defense_name = None
        optimizer, scheduler = self.load_optim_config_file(model)
        D_instance = eval(self.defense_method[0])  #############
        config_file_path = self.config_defense_param_xml_dir
        args = xmlparser(config_file_path)
        defense, defense_name = D_instance(model, device, optimizer, scheduler, **args), self.defense_method
        return device, model, defense, defense_name

    def get_parameter(self, config_file_path, keyWord):
        args = xmlparser(config_file_path)
        # print(keyWord)
        content = self.get_content_form_xml(keyWord, **args)
        return content

    def get_content_form_xml(self, keyword = None, **kwargs):
        # print(kwargs)
        content = kwargs.get(keyword)
        return content