# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-03
@LastEditTime: 2021-12-10

基于句子嵌入模型[universal-sentence-encoder-xling-many](https://tfhub.dev/google/universal-sentence-encoder-xling-many/1)的向量距离限制

暂时不使用，因为tf_sentencepiece不支持>=2.3.0版本的tensorflow
"""

import os
from typing import NoReturn, Any, Optional, Sequence

import torch
import tensorflow as tf

# https://stackoverflow.com/questions/62647139/tensorflow-hub-throwing-this-error-sentencepieceop-when-loading-the-link
import tensorflow_text  # noqa: F401
import tensorflow_hub as hub

from .sentence_encoder_base import SentenceEncoder
from ...misc import nlp_cache_dir
from ..._download_data import download_if_needed


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


__all__ = [
    "XlingManyUSE",
]


class XlingManyUSE(SentenceEncoder):
    """ """

    def __init__(
        self,
        path: Optional[str] = None,
        threshold: float = 0.8,
        metric: str = "angular",
        **kwargs: Any
    ) -> NoReturn:
        """ """
        self._path = path or os.path.join(
            nlp_cache_dir, "universal-sentence-encoder-xling-many_1"
        )
        if not os.path.exists(self._path):
            # raise NotImplementedError("暂不支持从 https://tfhub.dev/ 在线导入模型")
            # hub.load("https://tfhub.dev/google/universal-sentence-encoder-xling-many/1")
            download_if_needed(
                uri="universal-sentence-encoder-xling-many_1",
                source="aitesting",
                extract=True,
            )
        # Set up graph.
        g = tf.Graph()
        with g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            xling_8_embed = hub.Module(
                "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"
            )
            self.embedded_text = xling_8_embed(self.text_input)
            init_op = tf.group(
                [tf.global_variables_initializer(), tf.tables_initializer()]
            )
        g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g)
        self.session.run(init_op)

    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        """ """
        emb = self.session.run(
            self.embedded_text, feed_dict={self.text_input: sentences}
        ).numpy()
        return torch.from_numpy(emb)
