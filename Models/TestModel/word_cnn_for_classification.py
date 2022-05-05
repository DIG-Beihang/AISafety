# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2022-04-01

预置Word CNN分类模型
"""

import os
import glob
import json
from typing import NoReturn, Optional, List

import torch
from torch import nn as nn
from torch.nn import functional as F

from ._glove_embedding_layer import GloveEmbeddingLayer
from ..pytorch_model import PyTorchNLPVictimModel
from ..Tokenizers import GloveTokenizer
from utils.misc import nlp_cache_dir
from utils._download_data import download_if_needed


__all__ = [
    "VictimWordCNNForClassification",
]


class VictimWordCNNForClassification(PyTorchNLPVictimModel):
    """
    2分类或5分类模型，英文
    """

    __name__ = "VictimWordCNNForClassification"

    def __init__(self, path: Optional[str] = None) -> NoReturn:
        """ """
        self._path = path or os.path.join(nlp_cache_dir, "cnn-imdb")
        if not os.path.exists(self._path):
            # raise ValueError("暂不支持在线下载模型")
            download_if_needed(
                uri="cnn-imdb",
                source="aitesting",
                dst_dir=nlp_cache_dir,
                extract=True,
            )
        else:
            model = _WordCNNForClassification.from_pretrained(
                os.path.basename(self._path)
            )
        super().__init__(model, model.tokenizer)

    @property
    def path(self) -> str:
        return self._path

    def extra_repr_keys(self) -> List[str]:
        return ["path"]


class _WordCNNForClassification(nn.Module):
    """A convolutional neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    __name__ = "_WordCNNForClassification"
    _BUILTIN_MODELS = [
        "cnn-sst",
        "cnn-imdb",
    ]

    def __init__(
        self,
        hidden_size: int = 150,
        dropout: float = 0.3,
        num_labels: int = 2,
        max_seq_length: int = 128,
        model_path: Optional[str] = None,
        emb_layer_trainable: bool = True,
    ) -> NoReturn:
        """ """
        _config = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "model_path": None,
            "emb_layer_trainable": emb_layer_trainable,
        }
        if model_path:
            self = VictimWordCNNForClassification.from_pretrained(**_config)
            self._config["model_path"] = model_path
            self._config["architectures"] = self.__name__
            return

        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = CNNTextLayer(
            self.emb_layer.n_d, widths=[3, 4, 5], filters=hidden_size
        )
        d_out = 3 * hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

    def save_pretrained(self, output_path: str) -> NoReturn:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "_WordCNNForClassification":
        """Load trained Word CNN model by name or from path."""
        if name_or_path in _WordCNNForClassification._BUILTIN_MODELS:
            uri = f"models_v2/classification/{name_or_path.replace('-', '/')}"
            # path = os.path.join(nlp_cache_dir, *(uri.strip("/").split("/")))
            path = os.path.join(nlp_cache_dir, name_or_path)
            if not os.path.exists(path):
                download_if_needed(
                    uri, source="textattack", dst_dir=nlp_cache_dir, extract=True
                )
        else:
            path = name_or_path
        if not os.path.exists(path):
            raise ValueError(f"路径 {path} 不存在")

        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "WordCNNForClassification",
                "hidden_size": 150,
                "dropout": 0.3,
                "num_labels": 2,
                "max_seq_length": 128,
                "model_path": None,
                "emb_layer_trainable": True,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        emb = self.emb_layer(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.out(output)
        return pred

    def get_input_embeddings(self) -> nn.Module:
        return self.emb_layer.embedding


class CNNTextLayer(nn.Module):
    def __init__(
        self, n_in: int, widths: List[int] = [3, 4, 5], filters: int = 100
    ) -> NoReturn:
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]
        x = torch.cat(x, 1)
        return x


def load_cached_state_dict(model_folder_path: str) -> dict:
    # Take the first model matching the pattern *model.bin.
    model_path_list = glob.glob(os.path.join(model_folder_path, "*model.bin"))
    if not model_path_list:
        raise FileNotFoundError(
            f"model.bin not found in model folder {model_folder_path}."
        )
    model_path = model_path_list[0]
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    return state_dict
