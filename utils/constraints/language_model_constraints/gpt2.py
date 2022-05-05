# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2022-01-05
@LastEditTime: 2022-03-19

Language Model的限制条件：GPT2

目前中文模式下被调用可能报如下错误：
RuntimeError: CUDA error: device-side assert triggered
原因：中文模式使用的tokenizer可能会产出token_id等于30000的情况，而中文模型中的vocab_size是30000，
因此会IndexError，或者CUDA error: device-side assert triggered
解决方法：对产生的token_id使用clamp_max(30000-1)
或者使用try except
"""

from typing import NoReturn, Union, Sequence, List
import torch
from transformers import (  # noqa: F401
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForMaskedLM,
    AutoTokenizer,
    CpmTokenizer,
)

from .language_model_base import LanguageModelBase
from ...metrics.perplexity import _XLNetTokenizer
from ...attacked_text import AttackedText
from ...strings import normalize_language, LANGUAGE
from ...misc import default_device, nlp_cache_dir
from ..._download_data import download_if_needed


__all__ = [
    "GPT2",
]


class GPT2(LanguageModelBase):
    """A constraint based on the GPT-2 language model.
    from "Better Language Models and Their Implications"
    (openai.com/blog/better-language-models/)
    """

    def __init__(
        self, model_name: str = "gpt2", language: Union[LANGUAGE, str] = "zh", **kwargs
    ) -> NoReturn:
        """
        Args:
            model_name: id of GPT2 model
        """
        self._language = normalize_language(language)
        if self._language == LANGUAGE.ENGLISH:
            path = download_if_needed(
                uri="gpt2", source="aitesting", dst_dir=nlp_cache_dir, extract=True
            )
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        elif self._language == LANGUAGE.CHINESE:
            path = download_if_needed(
                uri="CPM-Generate-distill",
                source="aitesting",
                dst_dir=nlp_cache_dir,
                extract=True,
            )
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer = _XLNetTokenizer.from_pretrained(
                "mymusise/CPM-Generate-distill"
            )
            self.tokenizer = CpmTokenizer.from_pretrained(
                "mymusise/CPM-Generate-distill"
            )
            # 注意！
            # 这个tokenizer可能会产出token_id等于30000的情况，而模型中的vocab_size是30000，
            # 因此会IndexError，或者CUDA error: device-side assert triggered
            # 解决方法：对产生的token_id使用clamp_max(30000-1)
            # 或者使用try except
        try:
            self.model.to(default_device)
            self.device = default_device
        except Exception:
            self.model.to(torch.device("cpu"))
            self.device = torch.device("cpu")
        self.model.eval()
        super().__init__(**kwargs)

    def get_log_probs_at_index(
        self, text_list: Sequence[AttackedText], word_index: int
    ) -> Union[torch.Tensor, List[float]]:
        """
        Gets the probability of the word at index `word_index` according to GPT-2.
        Assumes that all items in `text_list` have the same prefix up until `word_index`.
        """
        prefix = text_list[0].text_until_word_index(word_index)
        print(prefix)

        # if not utils.has_letter(prefix):
        if len(prefix) == 0:
            # This language model perplexity is not defined with respect to
            # a word without a prefix. If the prefix is null, just return the
            # log-probability 0.0.
            return torch.zeros(len(text_list), dtype=torch.float)

        token_ids = self.tokenizer.encode(prefix)
        tokens_tensor = torch.tensor([token_ids])
        tokens_tensor = tokens_tensor.to(self.device)
        if self._language == LANGUAGE.CHINESE:
            tokens_tensor.clamp_max_(30000 - 1)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        predictions = outputs[0]

        probs = []
        for attacked_text in text_list:
            next_word_ids = self.tokenizer.encode(attacked_text.words[word_index])
            next_word_prob = predictions[0, -1, next_word_ids[0]]
            probs.append(next_word_prob)

        return probs
