# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-23
@LastEditTime: 2022-04-15
"""
import os
from typing import Union, Optional, Sequence, NoReturn, Dict

import pandas as pd
from bs4 import BeautifulSoup as BS

from .base import NLPDataset
from utils.strings import LANGUAGE


__all__ = [
    "DianPingTiny",
]


class DianPingTiny(NLPDataset):
    """ """

    __name__ = "DianPingTiny"

    def __init__(
        self,
        subsets: Optional[Union[Sequence[str], str]] = None,
        text_size: Optional[str] = None,
        remove_html: bool = True,
        max_len: Optional[int] = 512,
    ) -> NoReturn:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        if text_size:
            assert text_size.lower() in [
                "long",
                "xl",
            ]
            self._text_size = f"_{text_size.lower()}"
        else:
            self._text_size = ""
        fp = os.path.join(
            fp, "dianping_tiny", f"dianping_filtered{self._text_size}_tiny.csv.gz"
        )
        tot_ds = pd.read_csv(fp, lineterminator="\n")
        if subsets is None:
            _subsets = [
                "train",
                "test",
            ]
        elif isinstance(subsets, str):
            _subsets = [subsets]
        else:
            _subsets = subsets
        tot_ds = tot_ds[tot_ds.set.isin(_subsets)].reset_index(drop=True)
        if remove_html:
            tot_ds["text"] = tot_ds["text"].apply(
                lambda s: BS(s, features="lxml").get_text()
            )

        super().__init__(
            dataset=[(row["text"], row["label"]) for _, row in tot_ds.iterrows()],
            input_columns=[
                "text",
            ],
            label_map={idx + 1: idx for idx in range(2)},
            max_len=max_len,
        )
        self._name = self.__name__
        self._language = LANGUAGE.CHINESE

    def get_word_freq(self, use_log: bool = False) -> Dict[str, float]:
        """ """
        fp = os.path.dirname(os.path.abspath(__file__))
        cache_fp = os.path.join(
            fp,
            "dianping_tiny",
            f"dianping_filtered{self._text_size}_tiny_word_freq.csv.gz",
        )
        return super().get_word_freq(use_log=use_log, cache_fp=cache_fp, parallel=False)
