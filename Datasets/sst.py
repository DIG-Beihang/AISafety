# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-06
@LastEditTime: 2022-04-15
"""
import os
from typing import Union, Optional, Sequence, NoReturn, Dict

import pandas as pd

from .base import NLPDataset
from utils.strings import LANGUAGE


__all__ = [
    "SST",
]


class SST(NLPDataset):
    """ """

    __name__ = "SST"

    def __init__(
        self,
        subsets: Optional[Union[Sequence[str], str]] = None,
        binary: bool = True,
        max_len: Optional[int] = 512,
    ) -> NoReturn:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(fp, "sst", "sst.csv.gz")
        tot_ds = pd.read_csv(fp)
        if binary:
            tot_ds["label"] = tot_ds["label"].apply(lambda s: int(round(s)))
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
            dataset=[(row["sentence"], row["label"]) for _, row in tot_ds.iterrows()],
            input_columns=[
                "sentence",
            ],
            max_len=max_len,
        )
        self._name = self.__name__
        self._language = LANGUAGE.ENGLISH

    def get_word_freq(self, use_log: bool = False) -> Dict[str, float]:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        cache_fp = os.path.join(fp, "sst", "sst_word_freq.csv.gz")
        return super().get_word_freq(use_log=use_log, cache_fp=cache_fp, parallel=True)
