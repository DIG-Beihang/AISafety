# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-12-02
@LastEditTime: 2022-04-16

加载数据集命令行参数

"""

import importlib
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Dict, Any

from ..utils.strings import normalize_language, LANGUAGE


__all__ = [
    "DatasetArgs",
]


_BUILTIN_DATASETS = {
    "amazon": "AmazonReviewsZH",
    "dianping": "DianPingTiny",
    "imdb": "IMDBReviewsTiny",
    "jd_binary": "JDBinaryTiny",
    "jd_full": "JDFullTiny",
    "sst": "SST",
}

_NAME_MAPPING = {
    "amazon": "amazon_reviews_zh",
    "dianping": "dianping_tiny",
    "imdb": "imdb_reviews_tiny",
    "jd_binary": "jd_binary_tiny",
    "jd_full": "jd_full_tiny",
    "sst": "sst",
}


@dataclass
class DatasetArgs:
    """ """

    __name__ = "DatasetArgs"

    dataset: str = None
    subset: str = None
    max_len: int = 512

    @classmethod
    def _add_parser_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """ """
        default_obj = cls()
        ds_group = parser.add_argument_group()
        ds_group.add_argument(
            "-d",
            "--dataset",
            type=str,
            help="对抗攻击数据集",
            default=default_obj.dataset,
            dest="dataset",
        )
        ds_group.add_argument(
            "--subset",
            type=str,
            help="对抗攻击数据集子集名称",
            default=default_obj.subset,
            dest="subset",
        )
        ds_group.add_argument(
            "--max-len",
            type=int,
            help="对抗样本字符数目上限",
            default=default_obj.max_len,
            dest="max_len",
        )
        return parser

    @classmethod
    def _create_dataset_from_args(cls, args: Dict) -> Any:
        """ """
        obj = cls()
        obj.dataset = args.get("dataset")
        obj.subset = args.get("subset")
        obj.max_len = args.get("max_len")
        language = normalize_language(args.get("language"))
        if obj.dataset is None:
            if language == LANGUAGE.CHINESE:
                obj.dataset = "jd_binary"
            elif language == LANGUAGE.ENGLISH:
                obj.dataset = "sst"
            print(f"未指定数据集, 将加载默认数据集 {obj.dataset}")
        if obj.dataset in _BUILTIN_DATASETS:
            ds_cls = getattr(
                importlib.import_module(f"AISafety.Datasets.{_NAME_MAPPING[obj.dataset]}"),
                _BUILTIN_DATASETS[obj.dataset],
            )
        else:
            raise ValueError(f"{obj.dataset}不是内置数据集， 暂不支持自定义数据集")
        ds = ds_cls(subsets=obj.subset, max_len=obj.max_len)
        return ds
