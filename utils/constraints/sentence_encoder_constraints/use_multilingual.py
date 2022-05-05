# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-03
@LastEditTime: 2021-12-10

基于句子嵌入模型[universal-sentence-encoder-multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)的向量距离限制
"""

import os
from typing import NoReturn, Any, Optional, Sequence

import torch
import tensorflow as tf

# https://stackoverflow.com/questions/62647139/tensorflow-hub-throwing-this-error-sentencepieceop-when-loading-the-link
import tensorflow_text  # noqa: F401
import tensorflow_hub as hub

try:
    tf.config.gpu.set_per_process_memory_growth(True)
except Exception:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from .sentence_encoder_base import SentenceEncoderBase
from ...misc import nlp_cache_dir
from ..._download_data import download_if_needed


__all__ = [
    "MultilingualUSE",
]


class MultilingualUSE(SentenceEncoderBase):
    """ """

    __name__ = "MultilingualUSE"

    def __init__(
        self,
        path: Optional[str] = None,
        threshold: float = 0.8,
        metric: str = "angular",
        **kwargs: Any
    ) -> NoReturn:
        """ """
        self._path = path or os.path.join(
            nlp_cache_dir, "universal-sentence-encoder-multilingual_3"
        )
        if not os.path.exists(self._path):
            # raise NotImplementedError("暂不支持从 https://tfhub.dev/ 在线导入模型")
            # hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
            download_if_needed(
                uri="universal-sentence-encoder-multilingual_3",
                source="aitesting",
                extract=True,
            )
        self.model = hub.load(self._path)
        super().__init__(threshold=threshold, metric=metric, **kwargs)

    @torch.no_grad()
    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        """ """
        return torch.from_numpy(self.model(sentences).numpy())
        # return self.model(sentences)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state: dict) -> NoReturn:
        self.__dict__ = state
        self.model = hub.load(self._path)
