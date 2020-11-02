import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as Data

import torchvision
from torch.autograd import Variable


sys.path.append("{}/../".format(os.path.dirname(os.path.realpath(__file__))))
from EvalBox.Defense import *
from Models.UserModel import *
from EvalBox.Defense import *
from utils.EvalDataLoader import EvalDataset
from EvalBox.Analysis.Rebust_Defense import *
from utils.file_utils import read_dict_from_file


def main(args):
    print("run", args)
    dict_path = args.Dict_path
    dict_meta_batch = read_dict_from_file(dict_path)
    image_path = args.Data_path[0]
    label_path = args.Data_path[1]
    image_valid_path = []
    label_valid_path = []
    image_valid_path = args.Data_path[2]
    label_valid_path = args.Data_path[3]
    #####建立防御类对象#####
    r_a = Rebust_Defense(
        args.defense_method,
        image_path,
        label_path,
        image_valid_path,
        label_valid_path,
        1,
        "0",
        10,
        args.Scale_ImageSize,
        args.Crop_ImageSize,
        kwargs=args,
    )
    ######产生训练数据集#####
    dataset_train = r_a.gen_dataloader_train()
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size)
    ######产生测试数据集#####
    dataset_test = r_a.gen_dataloader_test()
    valid_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size)
    #####开始生成防御模型并保存
    acc, defense_enhanced_saver = r_a.gen_defense(train_loader, valid_loader)

    # r_a.load_model(args.model)
    # defense_enhanced_saver="./defense//PAT/CIFAR10_PAT_enhanced.pt"
    #####测试防御后的模型分类正确率，使用验证集的结果#####
    acc_valid = r_a.gen_valid_result(valid_loader, defense_enhanced_saver)

    print("defense enhanced acc:", acc_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Defense Model Generation")
    # common arguments
    parser.add_argument("--defense_method", type=str, nargs="*", default="EAT")
    parser.add_argument(
        "--Data_path",
        type=str,
        nargs="*",
        default=[
            "../Datasets/CIFAR_cln_data/cifar10_300_origin_inputs.npy",
            "../Datasets/CIFAR_cln_data/cifar10_300_origin_labels.npy",
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_inputs.npy",
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_labels.npy",
        ],
    )
    # "../test/Attack_generation/attack_param_FGSM_fgsm_01/FGSM_30_advs.npy", "../test/Attack_generation/attack_param_FGSM_fgsm_01/FGSM_30_advs_preds_labels.npy"])
    parser.add_argument(
        "--Dict_path", type=str, default="./dict_lists/cifar10_dict.txt"
    )
    parser.add_argument("--model", type=str, default="Models.UserModel.ResNet2")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # arguments for the particular attack
    parser.add_argument("--Scale_ImageSize", type=int, default=(32, 32))
    parser.add_argument("--Crop_ImageSize", type=int, default=(32, 32))
    parser.add_argument("--Enhanced_model_save_path", type=str, default="./defense/")
    parser.add_argument(
        "--config_defense_param_xml_dir", type=str, default="./defense/EAT/EAT.xml"
    )
    parser.add_argument(
        "--optim_config_dir", type=str, default="./defense/EAT/EAT_optim.xml"
    )
    parser.add_argument(
        "--config_model_dir_path", type=str, default="./defense/EAT/EAT_model.xml"
    )
    parser.add_argument("--data_type", type=str, default="CIFAR10")
    parser.add_argument(
        "--GPU_Config",
        type=str,
        # 数目，index设置
        default=["2", "0,1"],
    )
    arguments = parser.parse_args()
    main(args=arguments)
