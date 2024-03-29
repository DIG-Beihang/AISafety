# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-26
@LastEditTime: 2021-09-26

基于句子嵌入模型[InferSent](https://github.com/facebookresearch/InferSent)的向量距离限制
"""

import os
import io
import time
import zipfile
from typing import NoReturn, Any, Sequence, Tuple, Dict, List

import numpy as np
import torch
from torch import nn as nn

from .sentence_encoder_base import SentenceEncoderBase

# from ...word_embeddings import WordEmbedding
from ...misc import default_device
from ..._download_data import download_if_needed


__all__ = [
    "InferSent",
]


class InferSent(SentenceEncoderBase):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using InferSent."""

    MODEL_PATH = "infersent-encoder"
    WORD_EMBEDDING_PATH = "crawl-300d-2M.vec.zip"
    __name__ = "InferSent"

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        super().__init__(*args, **kwargs)
        self.model = self.get_infersent_model()
        self.model.to(self._device)

    def get_infersent_model(self) -> "InferSentModel":
        """Retrieves the InferSent model.

        Returns:
            The pretrained InferSent model.
        """
        infersent_version = 2
        model_folder_path = download_if_needed(InferSent.MODEL_PATH, source="aitesting")
        model_path = os.path.join(
            model_folder_path, f"infersent{infersent_version}.pkl"
        )
        params_model = {
            "bsize": 64,
            "word_emb_dim": 300,
            "enc_lstm_dim": 2048,
            "pool_type": "max",
            "dpout_model": 0.0,
            "version": infersent_version,
        }
        infersent = InferSentModel(params_model)
        infersent.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        word_embedding_path = download_if_needed(
            InferSent.WORD_EMBEDDING_PATH, source="aitesting", extract=False
        )
        infersent.set_w2v_path(word_embedding_path)
        infersent.build_vocab_k_words(K=100000)
        return infersent

    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        return self.model.encode(sentences, tokenize=True)


class InferSentModel(nn.Module):
    """ """

    __name__ = "InferSentModel"

    def __init__(self, config: dict) -> NoReturn:
        """ """
        super().__init__()
        self.bsize = config["bsize"]
        self.word_emb_dim = config["word_emb_dim"]
        self.enc_lstm_dim = config["enc_lstm_dim"]
        self.pool_type = config["pool_type"]
        self.dpout_model = config["dpout_model"]
        self.version = 1 if "version" not in config else config["version"]

        self.enc_lstm = nn.LSTM(
            self.word_emb_dim,
            self.enc_lstm_dim,
            1,
            bidirectional=True,
            dropout=self.dpout_model,
        )

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = "<s>"
            self.eos = "</s>"
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = "<p>"
            self.eos = "</p>"
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self) -> bool:
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple: Tuple[torch.Tensor, int]) -> torch.Tensor:
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = (
            torch.from_numpy(idx_sort).to(default_device)
            if self.is_cuda()
            else torch.from_numpy(idx_sort)
        )
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = (
            torch.from_numpy(idx_unsort).to(default_device)
            if self.is_cuda()
            else torch.from_numpy(idx_unsort)
        )
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1)
            if self.is_cuda():
                sent_len = sent_len.to(default_device)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path: str) -> NoReturn:
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences: Sequence[str], tokenize: bool = True) -> dict:
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ""
        word_dict[self.bos] = ""
        word_dict[self.eos] = ""
        return word_dict

    def get_w2v(self, word_dict: dict) -> Dict[str, np.ndarray]:
        assert hasattr(self, "w2v_path"), "w2v path not set"
        # create word_vec with w2v vectors
        word_vec = {}
        if self.w2v_path.endswith("zip"):
            with zipfile.ZipFile(self.w2v_path, "r") as zf:
                f = io.TextIOWrapper(
                    zf.open(os.path.basename(self.w2v_path).replace(".zip", "")),
                    encoding="utf-8",
                )
                for line in f:
                    word, vec = line.split(" ", 1)
                    if word in word_dict:
                        word_vec[word] = np.fromstring(vec, sep=" ")
        with open(self.w2v_path, encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=" ")
        print("Found %s(/%s) words with w2v vectors" % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K: int) -> Dict[str, np.ndarray]:
        assert hasattr(self, "w2v_path"), "w2v path not set"
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=" ")
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=" ")

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences: Sequence[str], tokenize: bool = True) -> NoReturn:
        assert hasattr(self, "w2v_path"), "w2v path not set"
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        # print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K: int) -> Dict[str, np.ndarray]:
        assert hasattr(self, "w2v_path"), "w2v path not set"
        self.word_vec = self.get_w2v_k(K)
        # print('Vocab size : %s' % (K))

    def update_vocab(self, sentences: Sequence[str], tokenize: bool = True) -> NoReturn:
        assert hasattr(self, "w2v_path"), "warning : w2v path not set"
        assert hasattr(self, "word_vec"), "build_vocab before updating it"
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print(
            "New vocab size : %s (added %s words)"
            % (len(self.word_vec), len(new_word_vec))
        )

    def get_batch(self, batch: Tuple[int, int, int]) -> torch.Tensor:
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s: str) -> List[str]:
        from nltk.tokenize import word_tokenize

        if self.moses_tok:
            s = " ".join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(
        self, sentences: Sequence[int], bsize: int, tokenize: bool, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sentences = [
            [self.bos] + s.split() + [self.eos]
            if not tokenize
            else [self.bos] + self.tokenize(s) + [self.eos]
            for s in sentences
        ]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings

                warnings.warn(
                    'No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..'
                    % (sentences[i], i)
                )
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print(
                "Nb words kept : %s/%s (%.1f%s)" % (n_wk, n_w, 100.0 * n_wk / n_w, "%")
            )

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(
        self,
        sentences: Sequence[str],
        bsize: int = 64,
        tokenize: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose
        )

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx : stidx + bsize])
            if self.is_cuda():
                batch = batch.to(default_device)
            with torch.no_grad():
                batch = (
                    self.forward((batch, lengths[stidx : stidx + bsize]))
                    .data.cpu()
                    .numpy()
                )
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print(
                "Speed : %.1f sentences/s (%s mode, bsize=%s)"
                % (
                    len(embeddings) / (time.time() - tic),
                    "gpu" if self.is_cuda() else "cpu",
                    bsize,
                )
            )
        return embeddings
