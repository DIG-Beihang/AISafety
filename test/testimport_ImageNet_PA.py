import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import PIL.Image as Image
import matplotlib.pyplot as plt

sys.path.append("{}/../".format(os.path.dirname(os.path.realpath(__file__))))
from EvalBox.Analysis.evaluation_base import Evaluation_Base

from EvalBox.Attack.AdvAttack import *

from utils.file_utils import read_dict_from_file
from torchvision.models import *
from EvalBox.Analysis.grand_CAM import *
from Models.TestModel.ResNet2 import resnet20_cifar
from Models.TestModel.AlexNet_modify import AlexNet_modify
from EvalBox.Analysis.Rebust_Eval import *

from utils.io_utils import SaveWithJson_Result
from utils.io_utils import update_current_status
from utils.io_utils import mkdir, get_label_lines, get_image_from_path


def main(args):
    print("run", args)
    image_path = args.Data_path[0]
    label_path = args.Data_path[1]
    image_origin_path = []
    label_origin_path = []
    image_origin_path = args.Data_path[2]
    label_origin_path = args.Data_path[3]
    attName = args.attack_method
    files = os.listdir("attack_param/" + attName[0])
    param_list = read_dict_from_file("attack_param/" + attName[0] + ".txt")
    cln_xs_npy = []
    cln_ys_npy = []
    target_pred_npy = []
    black_origin_outputs = []
    black_adv_outputs = []
    black_defense_origin_outputs = None
    black_defense_adv_outputs = None
    tensor_cln_ys = None
    tensor_cln_xs = None
    tensor_ys_targeted = None
    for f in files:
        args.attack_method = [attName[0], "", "attack_param/" + attName[0] + "/" + f]
        print(args.attack_method, args.evaluation_method)
        r_a = Rebust_Attack(
            args.attack_method,
            image_path,
            label_path,
            image_origin_path,
            label_origin_path,
            int(args.GPU_Config[0]),
            args.GPU_Config[1],
            10,
            args.Scale_ImageSize,
            args.Crop_ImageSize,
            args.model,
            args.model_dir,
            args.defense_model,
            args.model_defence_dir,
            args.data_type,
            args.IS_WHITE,
            args.IS_SAVE,
            args.IS_COMPARE_MODEL,
            args.IS_TARGETTED,
            args.save_path,
            args.save_method,
            args.black_Result_dir,
            args.batch_size,
        )
        adv_xs_npy = r_a.gen_attack_Samples()
        adv_ys_npy = r_a.gen_Attack_Preds(adv_xs_npy)
        black_adv_outputs, black_defense_adv_outputs = r_a.gen_Attack_Result(adv_xs_npy)
        (
            black_origin_outputs,
            black_defense_origin_outputs,
        ) = r_a.gen_Attack_Origin_Result()

        # 只在第一次for循环计算
        if cln_xs_npy == []:
            cln_xs_npy, cln_ys_npy, target_pred_npy = r_a.gen_origin_Samples()
            tensor_cln_ys = torch.from_numpy(np.array(cln_ys_npy))
            tensor_cln_xs = torch.from_numpy(np.array(cln_xs_npy))
            # 如果是目标攻击这个cln_ys 是 targets 的
            tensor_ys_targeted = torch.from_numpy(np.array(target_pred_npy))
        else:
            n = None
        device = r_a.device
        # 评估函数预处理，numpy转化成torch的tensor
        tensor_adv_xs = torch.from_numpy(adv_xs_npy)
        tensor_adv_ys = torch.from_numpy(adv_ys_npy)

        r_e = Rebust_Evaluate(
            tensor_adv_xs,
            tensor_cln_xs,
            tensor_cln_ys,
            tensor_adv_ys,
            tensor_ys_targeted,
            evaluation_method=args.evaluation_method,
            IS_PYTHORCH_WHITE=args.IS_PYTHORCH_WHITE,
            IS_COMPARE_MODEL=args.IS_COMPARE_MODEL,
            IS_TARGETTED=args.IS_TARGETTED,
        )
        # 获取模型信息
        model, defense_model = r_a.set_models()
        if args.IS_PYTHORCH_WHITE:
            r_e.get_models(model, defense_model)
            r_e.device = device
        # IS_PYTHORCH_WHITE=True BD，RGB，RIC这些是只能白盒攻击的评测算法,是单独模型
        if not args.IS_COMPARE_MODEL:
            rst = r_e.gen_evaluate(black_origin_outputs, black_adv_outputs)
        else:
            rst = r_e.gen_evaluate(
                black_origin_outputs,
                black_adv_outputs,
                black_defense_origin_outputs,
                black_defense_adv_outputs,
            )

        # log信息

        SaveWithJson_Result(
            args.save_visualization_base_path,
            "table_list",
            attName[0],
            f,
            args.evaluation_method,
            rst,
        )
        print("Evaluation output : ", rst)
        update_current_status(
            args.save_visualization_base_path,
            attName[0],
            100 * int(f.replace(".xml", "").split("_")[-1]) / len(files),
        )
        topk_show_list = [0]
        Save_Eval_Visualization_Result(
            attName,
            args.data_type,
            f,
            args.Dict_path,
            device,
            adv_xs_npy,
            args.save_visualization_base_path,
            args.IS_COMPARE_MODEL,
            args.model,
            args.defense_model,
            model,
            defense_model,
            args.CAM_layer,
            image_origin_path,
            label_path,
            black_adv_outputs,
            black_origin_outputs,
            black_defense_adv_outputs,
            black_defense_origin_outputs,
            topk_show_list=topk_show_list,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Attack Generation")
    # common arguments
    parser.add_argument("--attack_method", type=str, nargs="*", default=["FGSM"])
    parser.add_argument("--evaluation_method", type=str, default="ACAC")
    parser.add_argument(
        "--Data_path",
        type=str,
        nargs="*",
        default=[
            "../Datasets/ImageNet/images/",
            "../Datasets/ImageNet/val_1.txt",
            "../Datasets/ImageNet/images/",
            "../Datasets/ImageNet/val_1.txt",
        ],
    )
    # default = ["../Datasets/ImageCustom/images/", "../Datasets/ImageCustom/val_200.txt", "../Datasets/ImageCustom/images/",
    #      "../Datasets/ImageCustom/val_200.txt"])
    parser.add_argument(
        "--Dict_path", type=str, default="./dict_lists/ImageNet_12_dict.txt"
    )
    parser.add_argument("--defense_model", type=str, default="torchvision.models.vgg19")
    parser.add_argument(
        "--model",
        type=str,
        # default='Models.TestModel.AlexNet_modify')
        default="torchvision.models.vgg16",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        # default='../../1.pth')
        default="",
    )
    parser.add_argument(
        "--model_defence_dir",
        type=str,
        # default='../../1.pth')
        default="",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        # default = 'ImageNet')
        default=["ImageNet", "withoutNormalize"],
    )
    parser.add_argument("--IS_WHITE", type=bool, default=True)
    parser.add_argument("--IS_PYTHORCH_WHITE", type=bool, default=False)
    parser.add_argument("--IS_DOCKER_BLACK", type=bool, default=False)
    parser.add_argument("--black_Result_dir", type=str, default="..")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    # arguments for the particular attack
    parser.add_argument("--IS_SAVE", type=bool, default=False)
    parser.add_argument("--IS_COMPARE_MODEL", type=bool, default=False)
    parser.add_argument("--Scale_ImageSize", type=int, default=(375, 500))
    parser.add_argument("--Crop_ImageSize", type=int, default=(375, 500))
    parser.add_argument("--IS_TARGETTED", type=bool, default=False)
    parser.add_argument("--CAM_layer", type=int, default=28)
    parser.add_argument("--save_path", type=str, default="ImageNet_Attack_generation/")
    parser.add_argument("--save_method", type=str, default="ImageNet")
    parser.add_argument(
        "--GPU_Config",
        type=str,
        # 数目，index设置
        default=["2", "0,1"],
    )
    parser.add_argument("--save_visualization_base_path", type=str, default="./temp/")
    arguments = parser.parse_args()
    main(args=arguments)
