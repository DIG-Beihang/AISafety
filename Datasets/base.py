# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-16
@LastEditTime: 2022-04-15

Dataset类，用于载入数据进行模型评测，
主要基于TextAttack的实现
"""

import os
import re
import random
import math
import multiprocessing as mp
from string import punctuation
from collections import OrderedDict, Counter
from typing import List, Tuple, Optional, Dict, NoReturn, Sequence, Union, Iterable
from numbers import Real

import pandas as pd
from torch.utils.data import Dataset as TD
from datasets import (
    Dataset as HFD,
    NamedSplit as HFNS,
    load_dataset as HFD_load_dataset,
)
from zhon.hanzi import punctuation as zh_punctuation

from utils.strings import LANGUAGE, ReprMixin
from utils.attacked_text import AttackedText


__all__ = [
    "NLPDataset",
]


class NLPDataset(TD, ReprMixin):
    """ """

    __name__ = "NLPDataset"

    def __init__(
        self,
        dataset: List[tuple],
        input_columns: List[str] = ["text"],
        label_map: Optional[Dict[int, int]] = None,
        label_names: Optional[List[str]] = None,
        output_scale_factor: Optional[float] = None,
        shuffle: bool = False,
        max_len: Optional[int] = 512,
    ) -> NoReturn:
        """
        @param {
            dataset:
                A list of :obj:`(input, output)` pairs.
                If :obj:`input` consists of multiple fields (e.g. "premise" and "hypothesis" for SNLI),
                :obj:`input` must be of the form :obj:`(input_1, input_2, ...)` and :obj:`input_columns` parameter must be set.
                :obj:`output` can either be an integer representing labels for classification or a string for seq2seq tasks.
            input_columns:
                List of column names of inputs in order.
            label_map:
                Mapping if output labels of the dataset should be re-mapped. Useful if model was trained with a different label arrangement.
                For example, if dataset's arrangement is 0 for `Negative` and 1 for `Positive`, but model's label
                arrangement is 1 for `Negative` and 0 for `Positive`, passing :obj:`{0: 1, 1: 0}` will remap the dataset's label to match with model's arrangements.
                Could also be used to remap literal labels to numerical labels (e.g. :obj:`{"positive": 1, "negative": 0}`).
            label_names:
                List of label names in corresponding order (e.g. :obj:`["World", "Sports", "Business", "Sci/Tech"]` for AG-News dataset).
                If not set, labels will printed as is (e.g. "0", "1", ...). This should be set to :obj:`None` for non-classification datasets.
            output_scale_factor:
                Factor to divide ground-truth outputs by. Generally, TextAttack goal functions require model outputs between 0 and 1.
                Some datasets are regression tasks, in which case this is necessary.
            shuffle: Whether to shuffle the underlying dataset.
            max_len: Maximum length of input text.
        }
        @return: None
        """
        self._language = LANGUAGE.ENGLISH  # default language
        self._dataset = dataset
        self._name = None
        self.input_columns = input_columns
        self.label_map = label_map
        self.label_names = label_names
        if self.label_map and self.label_names:
            # If labels are remapped, the label names have to be remapped as well.
            self.label_names = [
                self.label_names[self.label_map[i]] for i in self.label_map
            ]
        self.shuffled = shuffle
        self.output_scale_factor = output_scale_factor

        if shuffle:
            random.shuffle(self._dataset)

        self.max_len = max_len

        self._word_freq = None
        self._word_log_freq = None

    def _format_as_dict(self, example: tuple) -> tuple:
        output = example[1]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        if isinstance(example[0], str):
            if len(self.input_columns) != 1:
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` and number of columns of actual input."
                )
            input_dict = OrderedDict(
                [(self.input_columns[0], self.clip_text(example[0]))]
            )
        else:
            if len(self.input_columns) != len(example[0]):
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` and number of columns of actual input."
                )
            input_dict = OrderedDict(
                [
                    (c, self.clip_text(example[0][i]))
                    for i, c in enumerate(self.input_columns)
                ]
            )
        return input_dict, output

    def shuffle(self) -> NoReturn:
        random.shuffle(self._dataset)
        self.shuffled = True

    def filter_by_labels_(self, labels_to_keep: Iterable) -> NoReturn:
        """Filter items by their labels for classification datasets. Performs
        in-place filtering.

        Args:
            labels_to_keep:
                Set, tuple, list, or iterable of integers representing labels.
        """
        if not isinstance(labels_to_keep, set):
            labels_to_keep = set(labels_to_keep)
        self._dataset = filter(lambda x: x[1] in labels_to_keep, self._dataset)

    def __getitem__(self, i: Union[slice, int]) -> Union[tuple, List[tuple]]:
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self._dataset[i]]

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._dataset)

    @staticmethod
    def from_huggingface_dataset(
        ds: Union[str, HFD], split: Optional[HFNS] = None, max_len: Optional[int] = 512
    ) -> "NLPDataset":
        """ """
        if isinstance(ds, str):
            _ds = HFD_load_dataset(ds, split=split)
        else:
            _ds = ds
        if isinstance(_ds.column_names, dict):
            sets = list(_ds.column_names.keys())
            column_names = _ds.column_names[sets[0]]
        else:
            sets = []
            column_names = _ds.column_names
        input_columns, output_column = _split_dataset_columns(column_names)

        if sets:
            ret_ds = NLPDataset(
                [
                    (_gen_input(row, input_columns), row[output_column])
                    for s in sets
                    for row in _ds[s]
                ],
                input_columns=input_columns,
                max_len=max_len,
            )
        else:
            ret_ds = NLPDataset(
                [(_gen_input(row, input_columns), row[output_column]) for row in _ds],
                input_columns=input_columns,
                max_len=max_len,
            )
            ret_ds._name = _ds.info.builder_name
        return ret_ds

    def get_word_freq(
        self,
        cache_fp: Optional[str] = None,
        use_log: bool = False,
        parallel: bool = False,
    ) -> Dict[str, float]:
        """ """
        if use_log and self._word_log_freq is not None:
            return self._word_log_freq
        elif not use_log and self._word_freq is not None:
            return self._word_freq
        if cache_fp is not None and os.path.exists(cache_fp):
            self._word_freq = pd.read_csv(cache_fp)
            self._word_freq = {
                w: f for w, f in zip(self._word_freq.word, self._word_freq.freq)
            }
            self._word_log_freq = {w: math.log(f) for w, f in self._word_freq.items()}
            if use_log:
                return self._word_log_freq
            return self._word_freq
        if parallel:
            with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
                processed = pool.starmap(
                    _get_word_freq_from_text,
                    iterable=[
                        (item[idx], self._language)
                        for item in self._dataset
                        for idx in range(len(item) - 1)
                    ],
                )
        else:
            processed = [
                _get_word_freq_from_text(item[idx], self._language)
                for item in self._dataset
                for idx in range(len(item) - 1)
            ]
        self._word_freq = Counter()
        for p in processed:
            self._word_freq += Counter(p)
        self._word_freq = dict(self._word_freq)
        self._word_log_freq = {w: math.log(f) for w, f in self._word_freq.items()}
        if cache_fp is not None:
            to_save = pd.DataFrame(
                list(self._word_freq.items()), columns=["word", "freq"]
            )
            to_save.to_csv(cache_fp, index=False, encoding="utf-8", compression="gzip")
            del to_save
        if use_log:
            return self._word_log_freq
        return self._word_freq

    @property
    def word_freq(self) -> Dict[str, Real]:
        """ """
        return self._word_freq

    @property
    def word_log_freq(self) -> Dict[str, Real]:
        """ """
        return self._word_log_freq

    def clip_text(self, text: str) -> str:
        """ """
        if self.max_len is None:
            return text
        inds = [
            m.start()
            for m in re.finditer(f"[{punctuation+zh_punctuation}]", text)
            if m.start() < self.max_len
        ]
        if len(inds) == 0:
            return text[: self.max_len]
        return text[: inds[-1]]

    @property
    def dataset_name(self) -> str:
        return self._name

    def extra_repr_keys(self) -> List[str]:
        if self.dataset_name is not None:
            return ["dataset_name"]
        return super().extra_repr_keys()


def _get_word_freq_from_text(text: str, language: LANGUAGE) -> Dict[str, float]:
    """ """
    t = AttackedText(language, text)
    return t.word_count


def _gen_input(row: dict, input_columns: Tuple[str]) -> Union[Tuple[str, ...], str]:
    """ """
    if len(input_columns) == 1:
        return row[input_columns[0]]
    return tuple(row[c] for c in input_columns)


def _split_dataset_columns(column_names: Sequence[str]) -> Tuple[Tuple[str, ...], str]:
    """Common schemas for datasets found in huggingface datasets hub."""
    _column_names = set(column_names)
    if {"premise", "hypothesis", "label"} <= _column_names:
        input_columns = ("premise", "hypothesis")
        output_column = "label"
    elif {"question", "sentence", "label"} <= _column_names:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"sentence1", "sentence2", "label"} <= _column_names:
        input_columns = ("sentence1", "sentence2")
        output_column = "label"
    elif {"question1", "question2", "label"} <= _column_names:
        input_columns = ("question1", "question2")
        output_column = "label"
    elif {"question", "sentence", "label"} <= _column_names:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"text", "label"} <= _column_names:
        input_columns = ("text",)
        output_column = "label"
    elif {"sentence", "label"} <= _column_names:
        input_columns = ("sentence",)
        output_column = "label"
    elif {"document", "summary"} <= _column_names:
        input_columns = ("document",)
        output_column = "summary"
    elif {"content", "summary"} <= _column_names:
        input_columns = ("content",)
        output_column = "summary"
    elif {"label", "review"} <= _column_names:
        input_columns = ("review",)
        output_column = "label"
    else:
        raise ValueError(
            f"Unsupported dataset column_names {_column_names}. Try passing your own `dataset_columns` argument."
        )

    return input_columns, output_column
