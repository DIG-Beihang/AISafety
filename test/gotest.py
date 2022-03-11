import argparse

import numpy as np
import torch
import yaml
from easydict import EasyDict

from EvalBox.Analysis.Rebust_Eval import Rebust_Attack, Rebust_Evaluate, Save_Eval_Visualization_Result
from utils.config import Config
from utils.io_utils import SaveWithJson_Result


def main(args):
    # 参数赋值
    if args.config.endswith('py'):
        cfg = Config.fromfile(args.config)
    elif args.config.endswith('yml') or args.config.endswith('yaml'):
        f = open(args.config, 'r', encoding='utf-8')
        c = f.read()
        cfg = yaml.load(c, Loader=yaml.FullLoader)
        cfg = EasyDict(cfg)
    else:
        print('unsupported config file type.')
        return
    image_path, label_path, image_origin_path, label_origin_path = cfg.datasets.test_path
    # 初始化中间变量
    cln_xs_npy = []
    tensor_cln_ys = None
    tensor_cln_xs = None
    tensor_ys_targeted = None
    # 根据不同的攻击参数，循环调用攻击过程
    for param in cfg.attack.params:
        # 攻击类对象
        r_a = Rebust_Attack(
            attack_method=[cfg.attack.type, param],
            sample_path=image_path,
            label_path=label_path,
            image_origin_path=image_origin_path,
            label_origin_path=label_origin_path,
            gpu_counts=cfg.gpu.nums,
            gpu_indexs=cfg.gpu.index,
            seed=0,
            Scale_ImageSize=cfg.datasets.augment.Crop_ImageSize,
            Crop_ImageSize=cfg.datasets.augment.Scale_ImageSize,
            model=cfg.model.path,
            model_dir=cfg.model.weights,
            defense_model=cfg.defense.model,
            model_defense_dir=cfg.defense.path,
            data_type=cfg.datasets.type,
            IS_WHITE=cfg.attack.IS_WHITE,
            IS_SAVE=cfg.result.IS_SAVE,
            IS_COMPARE_MODEL=cfg.attack.IS_COMPARE_MODEL,
            IS_TARGETTED=cfg.attack.IS_TARGETED,
            save_path=cfg.result.save_path,
            save_method=cfg.result.save_method,
            black_Result_dir=cfg.result.black_Result_dir,
            batch_size=cfg.datasets.batch_size,
        )
        # 获得攻击样本
        adv_xs_npy = r_a.gen_attack_Samples_by_conf()
        # 获得模型对攻击样本的预测值
        adv_ys_npy = r_a.gen_Attack_Preds(adv_xs_npy)
        # 比较模式下面，defense的也有值
        # 获得模型对攻击样本的预测概率值
        black_adv_outputs, black_defense_adv_outputs = r_a.gen_Attack_Result(adv_xs_npy)
        # 获得模型对原始样本的预测概率值
        black_origin_outputs, black_defense_origin_outputs = r_a.gen_Attack_Origin_Result()
        # 只在第一次for循环计算
        # 原始样本，原始样本标签，目标标签（非目标的时候和原始样本标签值一致）
        if not cln_xs_npy:
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
        # 新建的测评类的对象
        r_e = Rebust_Evaluate(
            tensor_adv_xs,
            tensor_cln_xs,
            tensor_cln_ys,
            tensor_adv_ys,
            tensor_ys_targeted,
            evaluation_method=cfg.evaluation.type,
            IS_PYTHORCH_WHITE=cfg.attack.IS_PYTORCH_WHITE,
            IS_COMPARE_MODEL=cfg.attack.IS_COMPARE_MODEL,
            IS_TARGETTED=cfg.attack.IS_TARGETED,
        )
        # 获取模型
        model, defense_model = r_a.set_models()
        # IS_PYTHORCH_WHITE=True BD，RGB，RIC这些是只能白盒攻击的评测算法,是单独模型
        if cfg.attack.IS_PYTORCH_WHITE:
            # 此时，测评对象需要模型的信息
            r_e.get_models(model, defense_model)
            r_e.device = device
        # 在单独一个模型的测评方法
        if not cfg.attack.IS_COMPARE_MODEL:
            rst = r_e.gen_evaluate(black_origin_outputs, black_adv_outputs)
        else:
            # 在两个模型比较的测评方法下
            rst = r_e.gen_evaluate(
                black_origin_outputs,
                black_adv_outputs,
                black_defense_origin_outputs,
                black_defense_adv_outputs,
            )
        # log信息
        #####保存测评信息，攻击方法，评测方法和对应的结果到指定目录文件下#####
        SaveWithJson_Result(
            cfg.result.save_visualization_base_path,
            "table_list",
            cfg.attack.type,
            Attack_file_name=param.name,
            evaluation_name=cfg.evaluation.type,
            value=rst
        )
        print("Evaluation output : ", rst)
        #####用户可根据自己需要选择展示的是哪些，以list形式传入#####
        topk_show_list = [0, 1]
        #####展示和保存可解释性分析的结果#####
        Save_Eval_Visualization_Result(
            cfg.attack.type,
            cfg.datasets.type,
            param.name,
            cfg.datasets.dict_path,
            device,
            adv_xs_npy,
            cfg.result.save_visualization_base_path,
            cfg.attack.IS_COMPARE_MODEL,
            cfg.model.path,
            cfg.defense.model,
            model,
            defense_model,
            cfg.datasets.CAM_layer,
            image_origin_path,
            label_path,
            black_adv_outputs,
            black_origin_outputs,
            black_defense_adv_outputs,
            black_defense_origin_outputs,
            topk_show_list=topk_show_list,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Attack and Evaluate Generation")
    parser.add_argument('config', help='you should type the config file path')
    arguments = parser.parse_args()
    main(args=arguments)
