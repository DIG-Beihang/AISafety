# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2022-07-14
@LastEditTime: 2022-07-14
"""

from typing import Union, Optional, Sequence, NoReturn, Dict

import pandas as pd

from .base import NLPDataset
from utils.strings import LANGUAGE
from utils.misc import nlp_cache_dir
from utils._download_data import download_if_needed


__all__ = [
    "Chinanews",
]


class Chinanews(NLPDataset):
    """ """

    __name__ = "Chinanews"

    def __init__(
        self,
        subsets: Optional[Union[Sequence[str], str]] = None,
        max_len: Optional[int] = 512,
    ) -> NoReturn:
        """ """
        fp = nlp_cache_dir / "chinanews" / "chinanews.csv.gz"
        if not fp.exists():
            download_if_needed(
                "chinanews.csv.gz",
                source="aitesting",
                cache_dir=nlp_cache_dir / "chinanews",
                extract=False,
            )
        tot_ds = pd.read_csv(fp, lineterminator="\n")
        tot_ds.loc[:, "label"] = tot_ds.label.map(int)
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

        super().__init__(
            dataset=[(row["content"], row["label"]) for _, row in tot_ds.iterrows()],
            input_columns=[
                "content",
            ],
            label_map={idx + 1: idx for idx in range(7)},
            label_names=[
                "mainland China politics",
                "Hong Kong - Macau politics",
                "International news",
                "financial news",
                "culture",
                "entertainment",
                "sports",
            ],
            max_len=max_len,
        )
        self._name = self.__name__
        self._language = LANGUAGE.CHINESE
        self.class_map = {  # start index converted from 1 to 0
            0: "mainland China politics",
            1: "Hong Kong - Macau politics",
            2: "International news",
            3: "financial news",
            4: "culture",
            5: "entertainment",
            6: "sports",
        }

    def get_word_freq(self, use_log: bool = False) -> Dict[str, float]:
        """ """
        cache_fp = nlp_cache_dir / "chinanews" / "chinanews_word_freq.csv.gz"
        return super().get_word_freq(use_log=use_log, cache_fp=cache_fp, parallel=False)
