import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision import utils as vutils
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from EvalBox.Analysis.evaluation_base import Evaluation_Base
from EvalBox.Evaluation import *
from EvalBox.UserEvaluation import *
from EvalBox.Attack import *
from utils.file_utils import read_dict_from_file
from torchvision.models import *
from EvalBox.Analysis.grand_CAM import *
from utils.file_utils import get_user_model, get_user_model_origin
from utils.io_utils import mkdir, get_label_lines, convertlist_to_numpy, \
    gen_attack_adv_save_path, save_json, read_json, load_json, analyze_json, output_value, dict_list_to_np
from utils.Attack_utils import *
from utils.io_utils import mkdir, get_label_lines, get_image_from_path
import json
#保存top-K的概率图，目前设定是前三
def show_bar_figure(attName, modelname, data_type, dict, adv_outputs, origin_outputs, path, index, datatype):
    mkdir(path)
    top_k = 3
    softmax_adv = F.softmax(adv_outputs[index], dim=0)
    softmaxsort_adv = torch.argsort(softmax_adv)
    softmax_oric = F.softmax(origin_outputs[index], dim=0)
    softmaxsort_oric = torch.argsort(softmax_oric)
    from matplotlib import pyplot as plt
    x = []
    y = []
    x2 = []
    y2 = []
    length = len(softmaxsort_adv.data.cpu().numpy())
    plt.clf()
    plt.rcParams['savefig.dpi'] = 300 #图片像素

    if "ImageNet" in data_type:
        plt.xticks(fontsize = 4)
        plt.figure(figsize = (16, 12))
        plt.xticks(rotation = -25)
        for i in range(top_k):
            ratiox = softmaxsort_adv.data.cpu().numpy()[length - 1 - i]
            ratiox_oric = softmaxsort_oric.data.cpu().numpy()[length - 1 - i]
            typename = dict[ratiox]
            typename2 = dict[ratiox_oric]
            x.append(typename)
            y.append(softmax_adv.data.cpu().numpy()[ratiox])
            x2.append(typename2)
            value_ratioy = softmax_oric.data.cpu().numpy()[ratiox_oric]
            y2.append(value_ratioy)
    else:
        plt.xticks(fontsize = 10)
        for i in range(top_k):
            ratiox = softmaxsort_adv.data.cpu().numpy()[length - 1 - i]
            ratiox_oric = softmaxsort_oric.data.cpu().numpy()[length - 1 - i]
            typename = dict[str(ratiox)]
            typename2 = dict[str(ratiox_oric)]
            x.append(typename)
            y.append(softmax_adv.data.cpu().numpy()[ratiox])
            x2.append(typename2)
            value_ratioy = softmax_oric.data.cpu().numpy()[ratiox_oric]
            y2.append(value_ratioy)
    plt.bar(x, y, color = 'r', align = 'center')  # adv_label
    plt.bar(x2, y2, color = 'g', align = 'center')  # the output of cln samples
    plt.ylabel('Prob value')
    plt.xlabel('Type')
    plt.savefig(path + "top_" + str(top_k) + "_" + str(index) + "_" + attName +"_"+modelname+ "_"+datatype + ".jpg")

#保存对抗样本和标签，原始样本和对应标签
def save_adv_result(adv_xs, adv_labels_numpy, class_num, \
                        device, attack_method, data_type, save_as_black_path, label_path, save_method, args_Attack_param):
    save_as_black_path = gen_attack_adv_save_path(save_as_black_path, args_Attack_param)
    mkdir(save_as_black_path)
    path_adv_xs = []
    path_adv_ys = []
    path_adv_xs_json = []
    path_adv_ys_json = []
    # 保存生成的攻击样
    if data_type == 'cifar10' or data_type == "cifar100" or save_method == ".npy":
        print('saving adv samples...')
        np.save(save_as_black_path + '/{}_{}_advs.npy'.format(attack_method, adv_xs.shape[0]), np.array(adv_xs))
        path_adv_xs = save_as_black_path + '/{}_{}_advs.npy'.format(attack_method, adv_xs.shape[0])
        path_adv_xs_json = save_as_black_path + '/{}_{}_advs.json'.format(attack_method, adv_xs.shape[0])
        save_json(path_adv_xs_json, adv_xs)
        print('saving adv labels...')
        # convert the adv_labels_numpy in vector type
        ys_save = []
        for i in range(adv_labels_numpy.shape[0]):
            y = np.zeros((1, class_num), dtype = np.uint8)
            y[0][adv_labels_numpy[i]] = 1
            ys_save.append(y[0])
        np.save(save_as_black_path + '/{}_{}_adv_preds_labels.npy'.format(attack_method, adv_xs.shape[0]), np.array(ys_save))
        path_adv_ys = save_as_black_path + '/{}_{}_adv_preds_labels.npy'.format(attack_method, adv_xs.shape[0])
        path_adv_ys_json = save_as_black_path + '/{}_{}_adv_preds_labels.json'.format(attack_method, adv_xs.shape[0])
        save_json(path_adv_ys_json, torch.from_numpy(np.array(ys_save)))

    else:
        # 保存成图片方式
        # 原始的一些图片和label信息
        print(label_path, "Saving in Image Model")
        image_names, label_list, number = get_label_lines(label_path)
        val_temp_path_name = "/Image/"
        image_temp_path_name = "/Image/Adv_Images/"
        save_val_path = save_as_black_path + val_temp_path_name
        mkdir(save_val_path)
        save_image_path = save_as_black_path + image_temp_path_name
        mkdir(save_image_path)
        val_file_name_adv_preds = save_val_path + "adv_preds_val_" + str(number) + '.txt'
        val_file_name_origins = save_val_path + "origins_val_" + str(number) + '.txt'
        with open(val_file_name_adv_preds, 'w') as f_adv:
            with open(val_file_name_origins, 'w') as f_origin:
                for i in range(adv_labels_numpy.shape[0]):
                    tensor_x = adv_xs[i]
                    path_name = "Adv_" + image_names[i]
                    resize_shape = (tensor_x.shape[1], tensor_x.shape[2])
                    adv_one_xs_resize = preprocess(image = tensor_x, resize = resize_shape, device = device)
                    vutils.save_image(adv_one_xs_resize.cpu(), save_image_path + path_name)
                    content_adv = path_name + " " + str(adv_labels_numpy[i]) + "\n"
                    f_adv.write(content_adv)
                    # content_origin = path_name + " " + str(cln_labels_numpy[i]) + "\n"
                    # f_origin.write(content_origin)
        f_adv.close()
        f_origin.close()
    return path_adv_xs, path_adv_ys, path_adv_xs_json, path_adv_ys_json

def save_numpy(inputs, path, discrube):
        np.save(path + '/{}_{}_outputs.npy'.format("black_predict_" + discrube, inputs.shape[0]), np.array(inputs))

class Rebust_Attack(Evaluation_Base):
    def __init__(self, attack_method = None, sample_path = None, label_path = None, image_origin_path = None, label_origin_path = None, 
                 gpu_counts = None, gpu_indexs = None, seed = None, Scale_ImageSize = None, Crop_ImageSize = None, 
                 model = None, model_dir = None, defense_model = None, model_defense_dir = None, data_type = None, IS_WHITE = None, 
                 IS_SAVE = None, IS_COMPARE_MODEL = None, IS_TARGETTED = None, save_path = None, save_method = None, black_Result_dir = None, batch_size = None):
        self._parse_params()
        self.defense_model_name = defense_model  # 'resnet20_cifar')
        self.model_name = model  # 'resnet20_cifar')
        self.data_type = data_type  # 'cifar10')
        self.model_dir = model_dir  # '../Models/TestModel/resnet20_cifar.pt')
        self.model_defense_dir = model_defense_dir  # '../Models/TestModel/resnet20_cifar.pt')
        self.IS_WHITE = IS_WHITE  # True)
        self.IS_SAVE = IS_SAVE  # False)
        self.IS_COMPARE_MODEL = IS_COMPARE_MODEL  # True)
        self.IS_TARGETTED = IS_TARGETTED  # False)
        self.batch_size = batch_size  # 64)
        self.Scale_ImageSize = Scale_ImageSize
        self.Crop_ImageSize = Crop_ImageSize
        self.save_as_black_path = save_path
        self.save_method = save_method
        self.black_Result_dir = black_Result_dir
        super(Rebust_Attack, self).__init__(attack_method, sample_path, label_path, image_origin_path, label_origin_path, 
                                            gpu_counts, gpu_indexs, seed, Scale_ImageSize, Crop_ImageSize, \
                                            model, model_dir, defense_model, model_defense_dir, data_type, IS_WHITE, \
                                            IS_SAVE, IS_COMPARE_MODEL, IS_TARGETTED, save_path, save_method, \
                                            black_Result_dir, batch_size)  # some model that in different setting


    def _parse_params(self):
        self.model = None          #自己生成对抗样本的原模型
        self.model_Defense = None   #比较模式下面，自己生成对抗样本的防御模型
        self.path_adv_xs = []
        self.path_adv_ys = []
        self.path_adv_xs_json = []
        self.path_adv_ys_json = []
        self.dataloader_origin = None
        self.dataset_origin = None
        self.dataloader = None
        self.dataset = None
    def set_models(self):
        return self.model, self.model_Defense
    def gen_origin_Samples(self):
        origin_xs_numpy = None
        origin_xs_numpy = self.get_origin_data(self.device, self.dataloader_origin)
        cln_ys_numpy, targeted_ys_numpy = self.get_origin_ys(self.device, self.dataloader, self.dataloader_origin)
        return origin_xs_numpy, cln_ys_numpy, targeted_ys_numpy
    def gen_origin_Result(self):
        origin_xs = self.get_origin_data(self.device, self.dataloader_origin)
        origin_xs_numpy = np.array(origin_xs)
        xs = torch.from_numpy(origin_xs_numpy)
        model = self.setting_model(self.model_dir, self.model_name, self.device, "origin")
        origin_outputs = self.outputs_eval(model, self.device, xs)
        return origin_outputs
    def gen_origin_defense_Result(self):
        origin_xs = self.get_origin_data(self.device, self.dataloader_origin)
        origin_xs_numpy = np.array(origin_xs)
        xs = torch.from_numpy(origin_xs_numpy)
        model_Defense = self.setting_model(self.model_defense_dir, self.defense_model_name, self.device, "defense")
        origin_outputs = self.outputs_eval(model_Defense, self.device, xs)
        return origin_outputs
    def gen_Attack_Origin_Result(self):
        black_defense_origin_outputs = None
        if not self.IS_COMPARE_MODEL:
            black_origin_outputs = self.gen_origin_Result()
        else:
            black_origin_outputs = self.gen_origin_Result()
            black_defense_origin_outputs = self.gen_origin_defense_Result()
        return black_origin_outputs, black_defense_origin_outputs
    def get_origin_data(self, device, dataloader):
        origin_xs_numpy = self.get_origin_sample(device, dataloader)
        return origin_xs_numpy
#分离单独的攻击模块
    def estimate_Attack_compare(self, device, adv_samples_numpy):
        adv_xs = torch.from_numpy(adv_samples_numpy)
        adv_outputs = self.outputs_eval(self.model, device, adv_xs)
        adv_outputs_numpy = np.array(adv_outputs)
        #防御模型需要的结果
        model_Defense = self.setting_model(self.model_defense_dir, self.defense_model_name, device, "defense")
        defense_adv_outputs_numpy = self.outputs_eval(model_Defense, device, adv_xs)
        return adv_outputs_numpy, defense_adv_outputs_numpy
    def estimate_Attack_uncompare(self, device, adv_samples_numpy):
        adv_xs = torch.from_numpy(adv_samples_numpy)
        #模型每次预测的时候需要再局部变量加载和设置
        model = self.setting_model(self.model_dir, self.model_name, device, "origin")
        adv_outputs = self.outputs_eval(model, device, adv_xs)
        adv_outputs_numpy = np.array(adv_outputs)
        return  adv_outputs_numpy
    def gen_Attack_Preds(self, adv_samples_numpy = None):
        adv_xs = torch.from_numpy(adv_samples_numpy).float()
        adv_preds = self.preds_eval(self.model, self.device, adv_xs)
        adv_labels_numpy = np.array(adv_preds).astype(int)
        return adv_labels_numpy
    def gen_Attack_Result(self, adv_samples_numpy = None):
        #生成了的对抗样本，目前是numpy的
        #两个模型做比较的话原始的样本不变，对抗样本使用的是在原模型上产生的
        #白盒产生的结果输出
        black_defense_adv_outputs = None
        if not self.IS_COMPARE_MODEL:
            black_adv_outputs = self.estimate_Attack_uncompare(self.device, adv_samples_numpy)
        else:
            black_adv_outputs, black_defense_adv_outputs = self.estimate_Attack_compare(self.device, adv_samples_numpy)
        return black_adv_outputs, black_defense_adv_outputs
    def load_black_Json_Result(self, dict_name):
        black_origin_outputs = None
        black_adv_outputs = None
        black_defense_origin_outputs = None
        black_defense_adv_outputs = None
        CD_dict = str(dict_name).split(".")[0]
        #print(CD_dict)
        if self.IS_COMPARE_MODEL:
            black_outputs_path = self.black_Result_dir
            json_content = load_json(black_outputs_path)
            analyze_json(json_content)
            model_content = output_value(json_content, "model")
            model_BDResult = output_value(model_content, "BDResult")
            model_CDResult = output_value(model_content, "CDResult")
            model_CDResult_dict = output_value(model_CDResult, CD_dict)
            black_origin_outputs = dict_list_to_np(model_BDResult)
            black_adv_outputs = dict_list_to_np(model_CDResult_dict)

            model_defense_content = output_value(json_content, "compare_model")
            model_defense_BDResult = output_value(model_defense_content, "BDResult")
            model_defense_CDResult = output_value(model_defense_content, "CDResult")
            model_defense_CDResult_dict = output_value(model_defense_CDResult, CD_dict)
            black_defense_origin_outputs = dict_list_to_np(model_defense_BDResult)
            black_defense_adv_outputs = dict_list_to_np(model_defense_CDResult_dict)

            return black_origin_outputs, black_adv_outputs, black_defense_origin_outputs, black_defense_adv_outputs
        else:
            black_outputs_path = self.black_Result_dir
            json_content = load_json(black_outputs_path)
            analyze_json(json_content)
            model_content = output_value(json_content, "model")
            model_BDResult = output_value(model_content, "BDResult")
            model_CDResult = output_value(model_content, "CDResult")
            model_CDResult_dict = output_value(model_CDResult, CD_dict)
            black_origin_outputs = dict_list_to_np(model_BDResult)
            black_adv_outputs = dict_list_to_np(model_CDResult_dict)
            return black_origin_outputs, black_adv_outputs

#产生和保存攻击样本和标签数据
    def gen_attack_Samples(self):
        model_dir = self.model_dir
        #这里用的是我们自己已经准备的一些已知网络结构的模型
        device, model, att, att_name = self.setting_device(model_dir, self.model_name)
#         model = self.get_model(model_dir, self.model_name, device)
        dataloader, dataset = self.setting_dataset(self.Scale_ImageSize, self.sample_path, self.label_path)
        dataloader_origin, dataset_origin = self.setting_dataset(self.Scale_ImageSize, self.image_origin_path, self.label_origin_path)
        
        self.model = model
        self.device = device
        self.dataloader_origin = dataloader_origin
        self.dataset_origin = dataset_origin
        self.dataloader = dataloader
        self.dataset = dataset
        IS_TARGETTED = self.IS_TARGETTED
        print("self.IS_SAVE", self.IS_SAVE)
        IS_SAVE = self.IS_SAVE
        # print('Loading the prepared  samples (nature inputs and corresponding labels) that will be attacked ...... ')
        print("self.IS_WHITE", self.IS_WHITE)
        if (self.IS_WHITE):
            class_num_type, adv_samples = self.white_eval(att, model, device, dataloader)
        else:
            class_num_type, adv_samples = self.black_eval(model, device, dataloader)
        
        # 攻击后的样本，黑盒下是迁移输入过来的
        adv_samples_numpy = np.copy(np.array(adv_samples))
        adv_xs = torch.from_numpy(np.copy(adv_samples_numpy))
        adv_preds = self.preds_eval(model, device, adv_xs)
        #攻击后经过模型计算出来的类别
        adv_labels_numpy = np.array(adv_preds).astype(int)
        #保存生成的攻击样本, 原始的不动，在别的地方单独保存，减少重复计算
        if IS_SAVE:
           self.path_adv_xs, self.path_adv_ys, self.path_adv_xs_json, self.path_adv_ys_json = save_adv_result(adv_xs, adv_labels_numpy, class_num_type, device, \
           self.attack_method[0], self.data_type, self.save_as_black_path, self.label_path, self.save_method, self.attack_method[2])
        self.gen_adv_save_result()
        #   对抗样本             原始样本      原始标签-groundtruth  对对抗样本预测的标签    如果目标攻击的目标标签    是否是目标非目标
        return adv_samples_numpy#, adv_samples_numpy.shape#, device, dataloader, dataloader_origin, att #该网络下对原始样本预测概率值, 对抗样本预测概率值（numpy格式）


    def gen_attack_Samples_by_conf(self):
        model_dir = self.model_dir
        # 这里用的是我们自己已经准备的一些已知网络结构的模型
        device, model, att, att_name = self.setting_device_by_conf(model_dir, self.model_name)
        #         model = self.get_model(model_dir, self.model_name, device)
        dataloader, dataset = self.setting_dataset(self.Scale_ImageSize, self.sample_path, self.label_path)
        dataloader_origin, dataset_origin = self.setting_dataset(self.Scale_ImageSize, self.image_origin_path,
                                                                 self.label_origin_path)

        self.model = model
        self.device = device
        self.dataloader_origin = dataloader_origin
        self.dataset_origin = dataset_origin
        self.dataloader = dataloader
        self.dataset = dataset
        IS_TARGETTED = self.IS_TARGETTED
        print("self.IS_SAVE", self.IS_SAVE)
        IS_SAVE = self.IS_SAVE
        # print('Loading the prepared  samples (nature inputs and corresponding labels) that will be attacked ...... ')
        print("self.IS_WHITE", self.IS_WHITE)
        if (self.IS_WHITE):
            class_num_type, adv_samples = self.white_eval(att, model, device, dataloader)
        else:
            class_num_type, adv_samples = self.black_eval(model, device, dataloader)

        # 攻击后的样本，黑盒下是迁移输入过来的
        adv_samples_numpy = np.copy(np.array(adv_samples))
        adv_xs = torch.from_numpy(np.copy(adv_samples_numpy))
        adv_preds = self.preds_eval(model, device, adv_xs)
        # 攻击后经过模型计算出来的类别
        adv_labels_numpy = np.array(adv_preds).astype(int)
        # 保存生成的攻击样本, 原始的不动，在别的地方单独保存，减少重复计算
        if IS_SAVE:
            self.path_adv_xs, self.path_adv_ys, self.path_adv_xs_json, self.path_adv_ys_json = save_adv_result(
                adv_xs, adv_labels_numpy, class_num_type, device, \
                self.attack_method[0], self.data_type, self.save_as_black_path, self.label_path, self.save_method,
                self.attack_method[2])
        self.gen_adv_save_result()
        #   对抗样本             原始样本      原始标签-groundtruth  对对抗样本预测的标签    如果目标攻击的目标标签    是否是目标非目标
        return adv_samples_numpy  # , adv_samples_numpy.shape#, device, dataloader, dataloader_origin, att #该网络下对原始样本预测概率值, 对抗样本预测概率值（numpy格式）


    def gen_adv_save_result(self):
        path_adv_xs = self.path_adv_xs
        path_adv_ys = self.path_adv_ys
        path_adv_xs_json = self.path_adv_xs_json
        path_adv_ys_json = self.path_adv_ys_json
        return path_adv_xs, path_adv_ys, path_adv_xs_json, path_adv_ys_json
    def get_adv_result(self):
        path_adv_xs = self.path_adv_xs
        path_adv_ys = self.path_adv_ys
        path_adv_xs_json = self.path_adv_xs_json
        path_adv_ys_json = self.path_adv_ys_json
        return path_adv_xs, path_adv_ys, path_adv_xs_json, path_adv_ys_json

class Rebust_Evaluate(object):
    def __init__(self, adv_xs = None, cln_xs = None, cln_ys = None, adv_ys = None, target_pred = None, device = None, \
                 outputs_origin = None, outputs_adv = None, defense_outputs_origin = None, defense_outputs_adv = None\
                 , evaluation_method = None, IS_PYTHORCH_WHITE = None, IS_COMPARE_MODEL = None, IS_TARGETTED = None):

        self.evaluation_method = evaluation_method
        self.IS_PYTHORCH_WHITE = IS_PYTHORCH_WHITE
        self.IS_COMPARE_MODEL = IS_COMPARE_MODEL  # True)
        self.IS_TARGETTED = IS_TARGETTED  # False)
        self._parse_params()
        self.device = device
        self.adv_xs = adv_xs
        self.cln_xs = cln_xs
        self.cln_ys = cln_ys
        self.adv_ys = adv_ys
        self.target_pred = target_pred

    def _parse_params(self):
        self.model = None  # 自己生成对抗样本的原模型
        self.model_Defense = None  # 比较模式下面，自己生成对抗样本的防御模型
    def get_models(self, model, model_defense):
        self.model = model
        self.model_Defense = model_defense
        return self.model, self.model_Defense
#黑盒模型预测后会得到这个  outputs_origin, outputs_defense
    def gen_evaluate(self, outputs_origin, outputs_adv, 
        defense_outputs_origin = None, defense_outputs_adv = None):
        acac_eval_origin = None
        device = self.device
        adv_xs = self.adv_xs
        cln_xs = self.cln_xs
        cln_ys = self.cln_ys
        adv_ys = self.adv_ys
        target_pred = self.target_pred

        IS_TARGETTED = self.IS_TARGETTED
        print("IS_COMPARE_MODEL", self.IS_COMPARE_MODEL)
        if self.IS_PYTHORCH_WHITE:
            print("IS_PYTHORCH_WHITE", self.IS_PYTHORCH_WHITE)
            if self.IS_COMPARE_MODEL:
                E_instance = eval(self.evaluation_method)
                acac_eval_origin, eva_name_origin = E_instance(outputs_origin, outputs_adv, defense_outputs_origin, 
                                                               defense_outputs_adv, device, self.model, self.model_defense), self.evaluation_method
                rst = acac_eval_origin.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_pred, IS_TARGETTED)
                return rst
            else:
                E_instance = eval(self.evaluation_method)
                acac_eval, eva_name = E_instance(outputs_origin, outputs_adv, device, self.model), self.evaluation_method
                rst = acac_eval.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_pred, IS_TARGETTED)
                return rst
        else:
             # 比较方式下的初始化，非局部变量
            if self.IS_COMPARE_MODEL:
                E_instance = eval(self.evaluation_method)
                acac_eval_origin, eva_name_origin = E_instance(outputs_origin, outputs_adv, defense_outputs_origin, defense_outputs_adv, device), self.evaluation_method
                rst = acac_eval_origin.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_pred, IS_TARGETTED)
                return rst
            else:
                E_instance = eval(str(self.evaluation_method))
                acac_eval, eva_name = E_instance(outputs_origin, outputs_adv, device), self.evaluation_method
                rst = acac_eval.evaluate(adv_xs, cln_xs, cln_ys, adv_ys, target_pred, IS_TARGETTED)
                return rst

class Rebust_Visual(object):
    def __init__(self, attName, modelname, dict, adv_outputs, origin_outputs, path, topk_number, datatype, topk_list):
        self.attName = attName
        self.modelname = modelname
        self.dict = dict
        self.adv_outputs = adv_outputs
        self.origin_outputs = origin_outputs
        self.path = path
        self.topk_number = topk_number
        self.datatype = datatype
        self.topk_list = topk_list
    def gen_visualization(self):
        attName = self.attName
        dict = self.dict
        adv_outputs = self.adv_outputs
        origin_outputs = self.origin_outputs
        path = self.path
        topk_number = self.topk_number
        datatype = self.datatype
        if self.topk_list is None:
            for i in range(topk_number):
                show_bar_figure(attName, self.modelname, self.datatype, dict, adv_outputs, origin_outputs, path, i, datatype)
        else:
            for index in self.topk_list:
                show_bar_figure(attName, self.modelname, self.datatype, dict, adv_outputs, origin_outputs, path, int(index), datatype)

def save_cam_result(image_origin_path, i, Crop_ImageSize, adv_xs_npy, device, \
                    model, CAM_layer, CAM_path, CAM_pathAttack, model_name, IS_COMPARE_MODEL, \
                    defense_model, defense_model_name):
    image, imgcv = get_image_from_path(image_origin_path, i, Crop_ImageSize, Crop_ImageSize)
    image_in = image
    imgcv_in = imgcv
    adv_xs_npy_reshape = np.transpose(adv_xs_npy[i], (1, 2, 0))
    adv_image = cv2.resize(adv_xs_npy_reshape, Crop_ImageSize)
    adv_image = np.ascontiguousarray(np.transpose(adv_image, (2, 0, 1)))
    use_cuda = False
    if not str(device) == "cpu":
        use_cuda = True

    get_CAM_ImageList(image_in, imgcv_in, model, use_cuda, CAM_layer, CAM_path + model_name + "_", i)
    get_CAM_ImageList(adv_image, imgcv_in, model, use_cuda, CAM_layer, CAM_pathAttack + model_name + "_", i)
    if IS_COMPARE_MODEL:
        get_CAM_ImageList(image_in, imgcv_in, defense_model, use_cuda, CAM_layer, CAM_path + defense_model_name + "_", 
                          i)
        get_CAM_ImageList(adv_image, imgcv_in, defense_model, use_cuda, CAM_layer, 
                          CAM_pathAttack + defense_model_name + "_", i)



def Save_Eval_Visualization_Result(attName, data_type, file_name, Dict_path, device, adv_xs_npy, save_base_path, \
                                   IS_COMPARE_MODEL, model_name, defense_model_name = None, \
                                   model = None, defense_model = None, CAM_layer = 28, \
                                   image_origin_path = None, label_path = None, \
                                   black_adv_outputs = None, black_origin_outputs = None, \
                                   black_defense_adv_outputs = None, black_defense_origin_outputs = None, topk_show_list = None):
    dict_path = Dict_path
    dict_meta_batch = read_dict_from_file(dict_path)
    if 'ImageNet' in data_type:
        _, _, paths_number = get_label_lines(label_path)
        path_index = file_name.split(".")[0]
        base_path = save_base_path + "/" + str(attName[0]) + "/" + path_index
        mkdir(base_path)
        topk_path = base_path + "/topk/"
        mkdir(topk_path)
        CAM_path = base_path + "/OriginSample_"
        CAM_pathAttack = base_path + "/AttackSample_"
        # 展示的分类概率的数目
        topk_number = paths_number
        adv_outputs = torch.from_numpy(black_adv_outputs)
        origin_outputs = torch.from_numpy(black_origin_outputs)

        r_v = Rebust_Visual(attName, model_name, dict_meta_batch, adv_outputs, origin_outputs, topk_path, topk_number, "ImageNet", topk_show_list)
        r_v.gen_visualization()

        if IS_COMPARE_MODEL:
            r_v_defense = Rebust_Visual(attName, defense_model_name, dict_meta_batch, adv_outputs, origin_outputs, topk_path, topk_number, "ImageNet", topk_show_list)
            r_v_defense.gen_visualization()

        Crop_ImageSize = (224, 224)
        #print(topk_show_list)
        if topk_show_list is None:
            for i in range(paths_number):
                save_cam_result(image_origin_path, i, Crop_ImageSize, adv_xs_npy, device, \
                                model, CAM_layer, CAM_path, CAM_pathAttack, model_name, IS_COMPARE_MODEL, \
                                defense_model, defense_model_name)
        else:
            for index in topk_show_list:
                save_cam_result(image_origin_path, int(index), Crop_ImageSize, adv_xs_npy, device, \
                                model, CAM_layer, CAM_path, CAM_pathAttack, model_name, IS_COMPARE_MODEL, \
                                defense_model, defense_model_name)
    elif 'cifar10' in data_type:
        topk_number = int(adv_xs_npy.shape[0] * 0.5)
        adv_outputs = torch.from_numpy(black_adv_outputs)
        origin_outputs = torch.from_numpy(black_origin_outputs)
        # 展示的分类概率的数目
        path_index = file_name.split(".")[0]
        base_path = save_base_path + "/"+str(attName[0]) + "/" + path_index
        mkdir(base_path)
        topk_path = base_path + "/topk/"
        mkdir(topk_path)
        r_v = Rebust_Visual(attName, model_name, dict_meta_batch, adv_outputs, origin_outputs, 
                            topk_path, topk_number, data_type, topk_show_list)
        r_v.gen_visualization()
        if IS_COMPARE_MODEL:
            topk_defense_number = int(adv_xs_npy.shape[0] * 0.5)
            adv_defense_outputs = torch.from_numpy(black_defense_adv_outputs)
            origin_defense_outputs = torch.from_numpy(black_defense_origin_outputs)
            r_v_defense = Rebust_Visual(attName, defense_model_name, dict_meta_batch, adv_defense_outputs, 
                                        origin_defense_outputs, 
                                        topk_path, topk_defense_number, data_type, topk_show_list)
            r_v_defense.gen_visualization()
    else:
        topk_number = int(adv_xs_npy.shape[0] * 0.5)
        adv_outputs = torch.from_numpy(black_adv_outputs)
        origin_outputs = torch.from_numpy(black_origin_outputs)
        # 展示的分类概率的数目
        path_index = file_name.split(".")[0]
        base_path = save_base_path + "/" + str(attName[0]) + "/" + path_index
        mkdir(base_path)
        topk_path = base_path + "/topk/"
        mkdir(topk_path)
        r_v = Rebust_Visual(attName, model_name, dict_meta_batch, adv_outputs, origin_outputs, 
                            topk_path, topk_number, data_type, topk_show_list)
        r_v.gen_visualization()
        if IS_COMPARE_MODEL:
            topk_defense_number = int(adv_xs_npy.shape[0] * 0.5)
            adv_defense_outputs = torch.from_numpy(black_defense_adv_outputs)
            origin_defense_outputs = torch.from_numpy(black_defense_origin_outputs)
            r_v_defense = Rebust_Visual(attName, defense_model_name, dict_meta_batch, adv_defense_outputs, 
                                        origin_defense_outputs, topk_path, topk_defense_number, data_type, topk_show_list)
            r_v_defense.gen_visualization()