# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-12-02
@LastEditTime: 2022-04-16

加载模型命令行参数

"""

import importlib
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Dict, Any

import transformers

from utils.strings import normalize_language, LANGUAGE
from .hf_model import HuggingFaceNLPVictimModel


__all__ = [
    "ModelArgs",
]


_TEST_MODELS = {
    "bert_amazon_zh": "VictimBERTAmazonZH",
    "roberta_chinanews": "VictimRoBERTaChinaNews",
    "roberta_dianping": "VictimRoBERTaDianPing",
    "roberta_ifeng": "VictimRoBERTaIFeng",
    "roberta_sst": "VictimRoBERTaSST",
}


@dataclass
class ModelArgs:
    """ """

    __name__ = "ModelArgs"

    model: str = None
    model_path: str = None

    @classmethod
    def _add_parser_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """ """
        default_obj = cls()
        model_group = parser.add_argument_group()
        model_group.add_argument(
            "-m",
            "--model",
            type=str,
            help=f"""被攻击模型名称，可以自定义，也可以从下面的列表{",".join(_TEST_MODELS.keys())}中选择，如果不指定，则使用默认模型""",
            default=default_obj.model,
            dest="model",
        )
        model_group.add_argument(
            "--model-path",
            type=str,
            help="被攻击模型路径",
            default=default_obj.model_path,
            dest="model_path",
        )
        return parser

    @classmethod
    def _create_model_from_args(cls, args: Dict) -> Any:
        """ """
        obj = cls()
        obj.model = args.get("model")
        obj.model_path = args.get("model_path")
        language = normalize_language(args.get("language"))
        if obj.model is None:
            if language == LANGUAGE.CHINESE:
                obj.model = "roberta_dianping"
            elif language == LANGUAGE.ENGLISH:
                obj.model = "roberta_sst"
            print(f"未指定模型, 将使用默认模型 {obj.model}")
        if obj.model in _TEST_MODELS:
            model_cls = getattr(
                importlib.import_module(f"Models.TestModel.{obj.model}"),
                _TEST_MODELS[obj.model],
            )
        elif obj.model.startswith("huggingface/"):
            model_name = obj.model.startswith("huggingface/", "")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = HuggingFaceNLPVictimModel(model, tokenizer)
            return model
        else:
            try:
                model_file, model_name = obj.model.split(".")
            except Exception:
                raise ValueError("自定义模型名称格式错误, 应该为 ``模型文件名.模型类名``")
            try:
                model_cls = getattr(
                    importlib.import_module(f"Models.UserModel.{model_file}"),
                    model_name,
                )
            except Exception:
                raise ValueError(f"未找到模型 {obj.model}")
        model = model_cls(obj.model_path)
        return model
