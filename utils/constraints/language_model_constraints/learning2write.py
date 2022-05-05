# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-12
@LastEditTime: 2022-03-19

Language Model的限制条件：Learning to Write
"""

import os
import gzip
import json
from typing import Sequence, NoReturn, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn

# import torchfile

from .language_model_base import LanguageModelBase
from ..._nn import RNNModel
from ...attacked_text import AttackedText
from ...strings_en import tokenize
from ...strings import ReprMixin
from ...misc import default_device, nlp_cache_dir
from ..._download_data import download_if_needed


__all__ = [
    "Learning2Write",
]


_CACHE_DIR = os.path.join(nlp_cache_dir, "learning2write")


class Learning2Write(LanguageModelBase):
    """A constraint based on the L2W language model.

    The RNN-based language model from "Learning to Write With Cooperative
    Discriminators" (Holtzman et al, 2018).

    https://arxiv.org/pdf/1805.06087.pdf

    https://github.com/windweller/l2w


    Reused by Jia et al., 2019, as a substitution for the Google 1-billion
    words language model (in a revised version the attack of Alzantot et al., 2018).

    https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689
    """

    __name__ = "Learning2Write"

    def __init__(
        self,
        window_size: int = 5,
        max_log_prob_diff: Optional[float] = None,
        compare_against_original: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """ """
        self.window_size = window_size
        self.query_handler = QueryHandler.load_model(
            _CACHE_DIR, kwargs.get("device", None)
        )
        super().__init__(max_log_prob_diff, compare_against_original)

    def get_log_probs_at_index(
        self, text_list: Sequence[AttackedText], word_index: int
    ) -> torch.Tensor:
        """Gets the probability of the word at index `word_index` according to the language model."""
        queries = []
        query_words = []
        for attacked_text in text_list:
            word = attacked_text.words[word_index]
            window_text = attacked_text.text_window_around_index(
                word_index, self.window_size
            )
            query = tokenize(window_text)
            queries.append(query)
            query_words.append(word)
        log_probs = self.query_handler.query(queries, query_words)
        return torch.tensor(log_probs)


class QueryHandler(ReprMixin):
    """ """

    __name__ = "QueryHandler"

    def __init__(
        self,
        model: nn.Module,
        word_to_idx: dict,
        mapto: torch.Tensor,
        device: torch.device,
    ) -> NoReturn:
        """ """
        self.model = model
        self.word_to_idx = word_to_idx
        self.mapto = mapto
        self.device = device

    def query(
        self,
        sentences: Sequence[str],
        swapped_words: Sequence[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Since we don't filter prefixes for OOV ahead of time, it's possible
        that some of them will have different lengths. When this is the case,
        we can't do RNN prediction in batch.

        This method _tries_ to do prediction in batch, and, when it
        fails, just does prediction sequentially and concatenates all of
        the results.
        """
        try:
            return self.try_query(sentences, swapped_words, batch_size=batch_size)
        except Exception:
            probs = []
            for s, w in zip(sentences, swapped_words):
                try:
                    probs.append(self.try_query([s], [w], batch_size=1)[0])
                except RuntimeError:
                    print(
                        "WARNING:  got runtime error trying languag emodel on language model w s/w",
                        s,
                        w,
                    )
                    probs.append(float("-inf"))
            return probs

    def try_query(
        self,
        sentences: Sequence[str],
        swapped_words: Sequence[str],
        batch_size: int = 32,
    ) -> List[int]:
        """ """
        # TODO use caching
        sentence_length = len(sentences[0])
        if any(len(s) != sentence_length for s in sentences):
            raise ValueError("Only same length batches are allowed")

        log_probs = []
        for start in range(0, len(sentences), batch_size):
            swapped_words_batch = swapped_words[
                start : min(len(sentences), start + batch_size)
            ]
            batch = sentences[start : min(len(sentences), start + batch_size)]
            raw_idx_list = [[] for i in range(sentence_length + 1)]
            for i, s in enumerate(batch):
                s = [word for word in s if word in self.word_to_idx]
                words = ["<S>"] + s
                word_idxs = [self.word_to_idx[w] for w in words]
                for t in range(sentence_length + 1):
                    if t < len(word_idxs):
                        raw_idx_list[t].append(word_idxs[t])
            orig_num_idxs = len(raw_idx_list)
            raw_idx_list = [x for x in raw_idx_list if len(x)]
            num_idxs_dropped = orig_num_idxs - len(raw_idx_list)
            all_raw_idxs = torch.tensor(
                raw_idx_list, device=self.device, dtype=torch.long
            )
            word_idxs = self.mapto[all_raw_idxs]
            hidden = self.model.init_hidden(len(batch))
            source = word_idxs[:-1, :]
            target = word_idxs[1:, :]
            if (not len(source)) or not len(hidden):
                return [float("-inf")] * len(batch)
            decode, hidden = self.model(source, hidden)
            decode = decode.view(sentence_length - num_idxs_dropped, len(batch), -1)
            for i in range(len(batch)):
                if swapped_words_batch[i] not in self.word_to_idx:
                    log_probs.append(float("-inf"))
                else:
                    log_probs.append(
                        sum(
                            [
                                decode[t, i, target[t, i]].item()
                                for t in range(sentence_length - num_idxs_dropped)
                            ]
                        )
                    )
        return log_probs

    @staticmethod
    def load_model(
        lm_folder_path: Optional[str] = None, device: Optional[torch.device] = None
    ) -> "QueryHandler":
        """ """
        _lm_folder_path = lm_folder_path or _CACHE_DIR
        download_if_needed(
            uri="learning2write",
            source="aitesting",
            dst_dir=nlp_cache_dir,
            extract=True,
        )
        _device = device or default_device
        # word_map = torchfile.load(os.path.join(_lm_folder_path, "word_map.th7"))
        # word_map = [w.decode("utf-8") for w in word_map]
        with gzip.open(os.path.join(_lm_folder_path, "word_map.json.gz"), "rt") as f:
            word_map = json.load(f)
        word_to_idx = {w: i for i, w in enumerate(word_map)}
        # word_freq = torchfile.load(
        #     os.path.join(os.path.join(_lm_folder_path, "word_freq.th7"))
        # )
        with gzip.open(os.path.join(_lm_folder_path, "word_freq.json.gz"), "rt") as f:
            word_freq = np.array(json.load(f)).astype(int)
        mapto = (
            torch.from_numpy(util_reverse(np.argsort(-word_freq))).long().to(_device)
        )

        model_file = open(os.path.join(_lm_folder_path, "lm-state-dict.pt"), "rb")

        model = RNNModel(
            "GRU",
            793471,
            256,
            2048,
            1,
            [4200, 35000, 180000, 793471],
            dropout=0.01,
            proj=True,
            lm1b=True,
        )

        model.load_state_dict(torch.load(model_file, map_location=_device))
        model.full = True  # Use real softmax--important!
        model.to(_device)
        model.eval()
        model_file.close()
        return QueryHandler(model, word_to_idx, mapto, _device)

    def extra_repr_keys(self) -> List[str]:
        return [
            "device",
            "model",
        ]


def util_reverse(item: Sequence) -> Sequence:
    new_item = np.zeros(len(item))
    for idx, val in enumerate(item):
        new_item[val] = idx
    return new_item
