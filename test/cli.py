# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-11-25
@LastEditTime: 2022-04-16

命令行运行文本评测模块

"""

import os
import sys
sys.path.append('../')
import json
import yaml
import argparse
import importlib
import warnings
from dataclasses import dataclass
from typing import Dict, NoReturn


if __name__ == "__main__" and __package__ is None:
    level = 1
    # https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass
    __package__ = ".".join(parent.parts[len(top.parts) :])
    importlib.import_module(__package__)  # won't be needed after that

from EvalBox.Attack.TextAttack.attack_eval import AttackEval
from EvalBox.Attack.TextAttack.attack_args import AttackArgs
from Models.model_args import ModelArgs
from Datasets.dataset_args import DatasetArgs
from utils.misc import module_dir, module_name


_ATTACK_RECIPES = {
    "a2t": "A2T",
    "bae": "BAE",
    "bert_attack": "BERTAttack",
    "checklist": "CheckList",
    "clare": "CLARE",
    "deep_word_bug": "DeepWordBug",
    "fd": "FD",
    "faster_genetic": "FasterGenetic",
    "genetic": "Genetic",
    "hotflip": "HotFlip",
    "iga": "IGA",
    "input_reduction": "InputReduction",
    "kuleshov": "Kuleshov",
    "pruthi": "Pruthi",
    "pso": "PSO",
    "pwws": "PWWS",
    "text_bugger": "TextBugger",
    "text_fooler": "TextFooler",
    "viper": "VIPER",
}
_RECOMMENDED_RECIPES = [
    "a2t",
    "bae",
    "bert_attack",
    "deep_word_bug",
    "fd",
    "hotflip",
    "pwws",
    "text_bugger",
    "text_fooler",
    "viper",
]


@dataclass
class CLIAttackArgs(AttackArgs, ModelArgs, DatasetArgs):
    """
    命令行参数
    """

    @classmethod
    def _add_parser_args(
        cls, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """ """
        parser = DatasetArgs._add_parser_args(parser)
        parser = ModelArgs._add_parser_args(parser)
        parser = AttackArgs._add_parser_args(parser)
        return parser


def get_args():
    """ """
    parser = argparse.ArgumentParser(
        description="Text Adversarial Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r", "--recipe", type=str, default="text_fooler", help="对抗攻击方法", dest="recipe"
    )
    parser.add_argument(
        "-l", "--language", type=str, default="zh", help="文本以及模型的语言", dest="language"
    )
    parser.add_argument(
        "--config-file", type=str, default=None, help="对抗攻击方法配置文件", dest="config_file"
    )
    parser.add_argument(
        "--time-out", type=float, default="5", help="单个样本处理时间上限，单位：分钟", dest="time_out"
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="忽略所有错误。默认关闭。关闭情况下，如果算法内部发生错误，则直接报错",
        dest="ignore_errors",
    )
    parser = CLIAttackArgs._add_parser_args(parser)

    args = vars(parser.parse_args())

    return args


def main(args: Dict) -> NoReturn:
    """ """
    args = get_args()

    model = ModelArgs._create_model_from_args(args)
    ds = DatasetArgs._create_dataset_from_args(args)

    time_out = args.pop("time_out")
    ignore_errors = args.pop("ignore_errors")

    recipe = args.pop("recipe")
    if recipe in _ATTACK_RECIPES:
        if recipe not in _RECOMMENDED_RECIPES:
            warnings.warn(f"{recipe} 并不推荐使用，可以更换为以下方法之一 {_RECOMMENDED_RECIPES}")
        recipe_cls = getattr(
            importlib.import_module(f"EvalBox.Attack.TextAttack.{recipe}"),
            _ATTACK_RECIPES[recipe],
        )
    else:
        raise ValueError(f"{recipe} 不在内置对抗攻击方法列表中, 暂不支持自定义对抗攻击方法")

    config_file = args.pop("config_file")
    if config_file is None:
        config_file = os.path.join(
            module_dir, "EvalBox", "Attack", "TextAttack", "default_configs.yml"
        )
        print("使用默认配置文件:", config_file)
    with open(config_file, "r") as f:
        if os.path.splitext(config_file)[-1] in [".yml", ".yaml"]:
            attacker_kwargs = yaml.safe_load(f)
        elif os.path.splitext(config_file)[-1] == ".json":
            attacker_kwargs = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件类型: {os.path.splitext(config_file)[-1]}")
    if recipe not in attacker_kwargs:
        raise ValueError(f"对抗攻击方法 {recipe} 不在配置文件中")
    attacker_kwargs = attacker_kwargs[recipe]

    attacker = recipe_cls(model=model, language=args.pop("language"), **attacker_kwargs)

    # print(f"args = {args}")

    ae = AttackEval(attacker, ds, CLIAttackArgs(**args))
    ae.attack_dataset(time_out=time_out, ignore_errors=ignore_errors)


if __name__ == "__main__":
    args = get_args()
    main(args)
