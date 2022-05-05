# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-02
@LastEditTime: 2022-04-01

预置模型，bert_amazon_zh
"""

import os
from typing import NoReturn, Optional, Union, Sequence, List

import torch
import numpy as np
import transformers

from ..hf_model import HuggingFaceNLPVictimModel
from utils.misc import nlp_cache_dir
from utils._download_data import download_if_needed


__all__ = [
    "VictimBERTAmazonZH",
]


class VictimBERTAmazonZH(HuggingFaceNLPVictimModel):
    """
    5 分类模型，中文，
    对商品评论进行评分预测，1-5分
    """

    __name__ = "VictimBERTAmazonZH"

    def __init__(self, path: Optional[str] = None) -> NoReturn:
        """ """
        self._path = path or os.path.join(nlp_cache_dir, "bert_amazon_reviews_zh")
        if not os.path.exists(self._path):
            # raise ValueError("暂不支持在线下载模型")
            download_if_needed(
                uri="bert_amazon_reviews_zh",
                source="aitesting",
                dst_dir=nlp_cache_dir,
                extract=True,
            )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self._path,
            num_labels=5,
            output_hidden_states=False,
        )
        tokenizer = transformers.BertTokenizer.from_pretrained(self._path)
        tokenizer.convert_id_to_word = tokenizer._convert_id_to_token
        super().__init__(model, tokenizer)
        self._pipeline = transformers.pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )

    @torch.no_grad()
    def predict(self, sentences: Union[str, Sequence[str]]) -> Union[int, List[int]]:
        """ """
        if isinstance(sentences, str):
            single_prediction = True
            pred = self([sentences])
        else:
            single_prediction = False
            pred = self(sentences)
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        pred = pred + 1  # indices to ratings
        if single_prediction:
            pred = pred[0]
        else:
            pred = pred.tolist()
        return pred

    def get_ratings(
        self, sentences: Union[str, Sequence[str]]
    ) -> Union[int, List[int]]:
        """ """
        return self.predict(sentences)

    @property
    def path(self) -> str:
        return self._path

    def extra_repr_keys(self) -> List[str]:
        return ["path"]
