# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-16
@LastEditTime: 2022-04-15
"""
import os
from typing import Union, Optional, Sequence, NoReturn, Dict

import pandas as pd

from .base import NLPDataset
from utils.strings import LANGUAGE


__all__ = [
    "AmazonReviewsZH",
]


class AmazonReviewsZH(NLPDataset):
    """ """

    __name__ = "AmazonReviewsZH"

    def __init__(
        self,
        subsets: Optional[Union[Sequence[str], str]] = None,
        max_len: Optional[int] = 512,
    ) -> NoReturn:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(fp, "amazon_reviews_zh", "amazon_reviews_zh.csv.gz")
        tot_ds = pd.read_csv(fp, lineterminator="\n")
        if subsets is None:
            _subsets = [
                "train",
                "validation",
                "test",
            ]
        elif isinstance(subsets, str):
            _subsets = [subsets]
        else:
            _subsets = subsets
        tot_ds = tot_ds[tot_ds.set.isin(_subsets)].reset_index(drop=True)

        super().__init__(
            dataset=[
                (row["review_body"], row["stars"]) for _, row in tot_ds.iterrows()
            ],
            input_columns=[
                "review_body",
            ],
            label_map={idx + 1: idx for idx in range(5)},
            max_len=max_len,
        )
        self._name = self.__name__
        self._language = LANGUAGE.CHINESE

    def get_word_freq(self, use_log: bool = False) -> Dict[str, float]:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        cache_fp = os.path.join(
            fp, "amazon_reviews_zh", "amazon_reviews_zh_word_freq.csv.gz"
        )
        return super().get_word_freq(use_log=use_log, cache_fp=cache_fp, parallel=False)
