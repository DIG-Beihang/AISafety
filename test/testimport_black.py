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
from EvalBox.Evaluation import *
from EvalBox.Attack import *
from utils.file_utils import read_dict_from_file
from torchvision.models import *
from EvalBox.Analysis.grand_CAM import *
from EvalBox.Analysis.Rebust_Eval import *
from utils.io_utils import SaveWithJson_Result
from utils.io_utils import update_current_status
from utils.io_utils import mkdir, get_label_lines, get_image_from_path


def main(arg1, arg2):
    print("run", arg1)
    dict_path = arg1.Dict_path
    dict_meta_batch = read_dict_from_file(dict_path)
    image_path = arg1.Data_path[0]
    label_path = arg1.Data_path[1]
    image_origin_path = []
    label_origin_path = []
    image_origin_path = arg1.Data_path[2]
    label_origin_path = arg1.Data_path[3]
    attName = arg1.attack_method
    files_O = os.listdir("attack_param/" + attName[0])
    param_list = read_dict_from_file("attack_param/" + attName[0] + ".txt")
    files = []
    if len(param_list) <= len(files_O):
        for filename in param_list:
            if str(filename) in files_O:
                files.append(str(filename))
    path_adv_xs_list = []
    path_adv_ys_list = []
    path_adv_xs_json_list = []
    path_adv_ys_json_list = []
    #####如果只需要生成攻击样本和攻击预测值，用于黑盒上使用#####
    if arg1.ONLY_GENRATE_BLACK_SAMPLE:
        for f in files:
            arg1.attack_method = [
                attName[0],
                "",
                "attack_param/" + attName[0] + "/" + f,
            ]
            print(arg1.attack_method)
            #####新建的攻击类的对象#####
            r_a = Rebust_Attack(
                arg1.attack_method,
                image_path,
                label_path,
                image_origin_path,
                label_origin_path,
                int(arg1.GPU_Config[0]),
                arg1.GPU_Config[1],
                10,
                arg1.Scale_ImageSize,
                arg1.Crop_ImageSize,
                arg1.model,
                arg1.model_dir,
                arg1.defense_model,
                arg1.model_defence_dir,
                arg1.data_type,
                arg1.IS_WHITE,
                arg1.IS_SAVE,
                arg1.IS_COMPARE_MODEL,
                arg1.IS_TARGETTED,
                arg1.save_path,
                arg1.save_method,
                arg1.black_Result_dir,
                arg1.batch_size,
            )
            #####产生攻击的结果#####
            r_a.gen_Attack_Result()
            #####产生的用于黑盒的攻击样本，并给出路径#####
            (
                path_adv_xs,
                path_adv_ys,
                path_adv_xs_json,
                path_adv_ys_json,
            ) = r_a.gen_adv_save_result()
            path_adv_xs_list.append(path_adv_xs)
            path_adv_ys_list.append(path_adv_ys)
            path_adv_xs_json_list.append(path_adv_xs_json)
            path_adv_ys_json_list.append(path_adv_ys_json)
        # print(path_adv_xs_list,path_adv_ys_list,path_adv_xs_json_list,path_adv_ys_json_list)
        return (
            path_adv_xs_list,
            path_adv_ys_list,
            path_adv_xs_json_list,
            path_adv_ys_json_list,
        )
    #####如果不是光生成攻击样本，而是用黑盒的预测结果来进行评测#####
    elif not arg1.ONLY_GENRATE_BLACK_SAMPLE and arg1.IS_DOCKER_BLACK:
        cln_xs_npy = []
        cln_ys_npy = []
        target_ys_npy = []
        black_origin_outputs = []
        black_adv_outputs = []
        black_defense_origin_outputs = None
        black_defense_adv_outputs = None
        tensor_cln_ys = None
        tensor_cln_xs = None
        tensor_ys_targeted = None
        #####设置使用的攻击算法的参数文件位置#####
        for f in files:
            arg1.attack_method = [
                attName[0],
                "",
                "attack_param/" + attName[0] + "/" + f,
            ]
            print(arg1.attack_method, f)
            #####新建的攻击类的对象#####
            r_a = Rebust_Attack(
                arg1.attack_method,
                image_path,
                label_path,
                image_origin_path,
                label_origin_path,
                int(arg1.GPU_Config[0]),
                arg1.GPU_Config[1],
                10,
                arg1.Scale_ImageSize,
                arg1.Crop_ImageSize,
                arg1.model,
                arg1.model_dir,
                arg1.defense_model,
                arg1.model_defence_dir,
                arg1.data_type,
                arg1.IS_WHITE,
                arg1.IS_SAVE,
                arg1.IS_COMPARE_MODEL,
                arg1.IS_TARGETTED,
                arg1.save_path,
                arg1.save_method,
                arg1.black_Result_dir,
                arg1.batch_size,
            )
            #####获得攻击样本#####
            adv_xs_npy = r_a.gen_attack_Samples()
            #####获得自有模型对攻击样本的预测值#####，实际不使用
            adv_ys_npy = r_a.gen_Attack_Preds(adv_xs_npy)
            # 只在第一次for循环计算
            #####原始样本，原始样本标签，目标标签（非目标的时候和原始样本标签值一致）#####
            if cln_xs_npy == []:
                cln_xs_npy, cln_ys_npy, target_ys_npy = r_a.gen_origin_Samples()
                tensor_cln_ys = torch.from_numpy(np.array(cln_ys_npy))
                tensor_cln_xs = torch.from_numpy(np.array(cln_xs_npy))
                # 如果是目标攻击这个cln_ys 是 targets 的
                tensor_ys_targeted = torch.from_numpy(np.array(target_ys_npy))
            else:
                n = None
            # black_origin_outputs是模型在黑盒下对原始的样本数据预测的结果，black_adv_outputs是模型在黑盒下对对抗样本的数据预测的结果
            # 如果是对比方式下，black_defense_origin_outputs是防御模型在黑盒下对原始的样本数据预测的结果，black_defense_adv_outputs是防御模型在黑盒下对对抗样本的数据预测的结果
            if not arg1.IS_COMPARE_MODEL:
                black_origin_outputs, black_adv_outputs = r_a.load_black_Json_Result(f)
            else:
                (
                    black_origin_outputs,
                    black_adv_outputs,
                    black_defense_origin_outputs,
                    black_defense_adv_outputs,
                ) = r_a.load_black_Json_Result(f)
            device = r_a.device
            # 评估函数预处理，numpy转化成torch的tensor
            tensor_adv_xs = torch.from_numpy(adv_xs_npy)
            tensor_adv_ys = torch.from_numpy(adv_ys_npy)

            #####新建的测评类的对象#####
            r_e = Rebust_Evaluate(
                tensor_adv_xs,
                tensor_cln_xs,
                tensor_cln_ys,
                tensor_adv_ys,
                tensor_ys_targeted,
                evaluation_method=arg2.evaluation_method,
                IS_PYTHORCH_WHITE=arg2.IS_PYTHORCH_WHITE,
                IS_COMPARE_MODEL=arg2.IS_COMPARE_MODEL,
                IS_TARGETTED=arg2.IS_TARGETTED,
            )

            model, defense_model = r_a.set_models()
            ######IS_PYTHORCH_WHITE=True BD，RGB，RIC这些是只能白盒攻击的评测算法，默认黑盒下面不使用这三个评测方法，IS_PYTHORCH_WHITE设置False#####
            if arg2.IS_PYTHORCH_WHITE:
                r_e.get_models(model, defense_model)
                r_e.device = device
            #####在单独一个模型的测评方法#####
            if not arg2.IS_COMPARE_MODEL:
                rst = r_e.gen_evaluate(black_origin_outputs, black_adv_outputs)
            else:
                #####在两个模型比较的测评方法下#####
                rst = r_e.gen_evaluate(
                    black_origin_outputs,
                    black_adv_outputs,
                    black_defense_origin_outputs,
                    black_defense_adv_outputs,
                )

            # log信息
            #####保存测评信息，攻击方法，评测方法和对应的结果到指定目录文件下#####
            SaveWithJson_Result(
                arg1.save_visualization_base_path,
                "table_list",
                attName[0],
                f,
                arg2.evaluation_method,
                rst,
            )
            print("Evaluation output : ", rst)
            update_current_status(
                arg1.save_visualization_base_path,
                attName[0],
                100 * int(f.replace(".xml", "").split("_")[-1]) / len(files),
            )

            topk_show_list = None
            #####用户可根据自己需要选择展示的是哪些，以list形式传入#####
            # topk_show_list=[0,1,3]
            #####展示和保存可解释性分析的结果#####
            Save_Eval_Visualization_Result(
                attName,
                arg1.data_type,
                f,
                arg1.Dict_path,
                device,
                adv_xs_npy,
                arg1.save_visualization_base_path,
                arg1.IS_COMPARE_MODEL,
                arg1.model,
                arg1.defense_model,
                model,
                defense_model,
                arg2.CAM_layer,
                image_origin_path,
                label_path,
                black_adv_outputs,
                black_origin_outputs,
                black_defense_adv_outputs,
                black_defense_origin_outputs,
                topk_show_list=topk_show_list,
            )


if __name__ == "__main__":
    parser1 = argparse.ArgumentParser(description="The Attack Generation")
    parser2 = argparse.ArgumentParser(description="The Evaluate Generation")
    # common arguments
    parser1.add_argument("--attack_method", type=str, nargs="*", default=["FGSM"])
    parser1.add_argument(
        "--Data_path",
        type=str,
        nargs="*",
        default=[
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_inputs.npy",
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_labels.npy",
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_inputs.npy",
            "../Datasets/CIFAR_cln_data/cifar10_30_origin_labels.npy",
        ],
    )
    parser1.add_argument(
        "--Dict_path", type=str, default="./dict_lists/cifar10_dict.txt"
    )
    parser1.add_argument(
        "--defense_model", type=str, default="Models.UserModel.ResNet2"
    )
    parser1.add_argument("--model", type=str, default="Models.UserModel.FP_resnet")
    parser1.add_argument(
        "--model_dir", type=str, default="../Models/weights/FP_ResNet20.th"
    )
    parser1.add_argument(
        "--model_defence_dir", type=str, default="../Models/weights/resnet20_cifar.pt"
    )
    parser1.add_argument("--IS_COMPARE_MODEL", type=bool, default=False)
    parser1.add_argument("--IS_TARGETTED", type=bool, default=False)
    parser1.add_argument("--IS_WHITE", type=bool, default=True)
    parser1.add_argument("--IS_PYTHORCH_WHITE", type=bool, default=False)
    parser1.add_argument("--IS_DOCKER_BLACK", type=bool, default=True)
    parser1.add_argument("--ONLY_GENRATE_BLACK_SAMPLE", type=bool, default=False)
    parser1.add_argument("--IS_SAVE", type=bool, default=True)
    parser1.add_argument(
        "--black_Result_dir", type=str, default="../Datasets/adv_data/zjx.json"
    )
    parser1.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser1.add_argument("--Scale_ImageSize", type=int, default=(32, 32))
    parser1.add_argument("--Crop_ImageSize", type=int, default=(32, 32))
    parser1.add_argument("--save_path", type=str, default="./Attack_generation/")
    parser1.add_argument("--save_method", type=str, default=".npy")
    parser1.add_argument(
        "--GPU_Config",
        type=str,
        # 名称，数目，index设置
        default=["2", "0,1"],
    )
    parser1.add_argument("--data_type", type=str, default="cifar10")
    parser1.add_argument("--save_visualization_base_path", type=str, default="./temp/")
    parser2.add_argument("--evaluation_method", type=str, default="ACC")
    parser2.add_argument("--IS_COMPARE_MODEL", type=bool, default=False)
    parser2.add_argument("--IS_PYTHORCH_WHITE", type=bool, default=False)
    parser2.add_argument("--IS_TARGETTED", type=bool, default=False)
    parser2.add_argument("--CAM_layer", type=int, default=12)
    arguments1 = parser1.parse_args()
    arguments2 = parser2.parse_args()
    main(arg1=arguments1, arg2=arguments2)
