# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-26
@LastEditTime: 2021-09-26

基于句子嵌入模型[BERT](https://huggingface.co/sentence-transformers)的向量距离限制
"""

import os
from typing import NoReturn, Any, Optional, Union

import torch
import sentence_transformers

from .sentence_encoder_base import SentenceEncoderBase
from ...strings import LANGUAGE, normalize_language
from ...misc import nlp_cache_dir
from ..._download_data import download_if_needed


__all__ = ["BERT"]


class BERT(SentenceEncoderBase):
    """
    Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using BERT, trained on NLI data, and
    fine-tuned on the STS benchmark dataset.
    """

    __name__ = "BERT"

    def __init__(
        self,
        threshold: float = 0.7,
        metric: str = "cosine",
        language: Union[str, LANGUAGE] = "en",
        path_or_name: Optional[str] = None,
        **kwargs: Any,
    ) -> NoReturn:
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self._language = normalize_language(language)
        if not path_or_name:
            if self._language == LANGUAGE.ENGLISH:
                path_or_name = "bert-base-nli-stsb-mean-tokens"
            elif self._language == LANGUAGE.CHINESE:
                raise ValueError
        if os.path.exists(path_or_name):
            self.model = sentence_transformers.SentenceTransformer(path_or_name)
        elif os.path.exists(os.path.join(nlp_cache_dir, path_or_name)):
            self.model = sentence_transformers.SentenceTransformer(
                os.path.join(nlp_cache_dir, path_or_name)
            )
        else:
            path = download_if_needed(
                path_or_name, source="aitesting", dst_dir=nlp_cache_dir
            )
            self.model = sentence_transformers.SentenceTransformer(path)
        try:
            self.model.to(self._device)
        except Exception:
            pass

    def encode(self, sentences) -> torch.Tensor:
        """ """
        return self.model.encode(sentences)
