# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-22
@LastEditTime: 2021-09-22

perplexity, 语言混乱程度度量，为 fluency，语言流畅程度，的相反度量
基于 TextAttack 的 Perplexity进行实现，参考了 OpenAttack 的 GPT2LM, GPT2LMCH

中文暂时有问题
"""

from typing import NoReturn, Union, Sequence

import torch
from transformers import (  # noqa: F401
    TFGPT2LMHeadModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForMaskedLM,
    AutoTokenizer,
    XLNetTokenizer,
)
import jieba

from .base import Metric
from ..misc import default_device, nlp_cache_dir
from .._download_data import download_if_needed
from ..strings import normalize_language, LANGUAGE
from ...EvalBox.Attack.attack_result import (
    GoalFunctionResult,
    FailedAttackResult,
    SkippedAttackResult,
)


__all__ = [
    "Perplexity",
]


class Perplexity(Metric):
    """
    Calculates average Perplexity on all successfull attacks using a pre-trained small GPT-2 model
    """

    __name__ = "Perplexity"

    def __init__(
        self, language: Union[str, LANGUAGE], model_name: str = "gpt2"
    ) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self.all_metrics = {}
        self.original_candidates = []
        self.successful_candidates = []
        if self._language == LANGUAGE.ENGLISH:
            path = download_if_needed(
                uri="gpt2", source="aitesting", dst_dir=nlp_cache_dir, extract=True
            )
            self.ppl_model = GPT2LMHeadModel.from_pretrained(path)
            self.ppl_tokenizer = GPT2Tokenizer.from_pretrained(path)
        elif self._language == LANGUAGE.CHINESE:
            path = download_if_needed(
                uri="CPM-Generate-distill",
                source="aitesting",
                dst_dir=nlp_cache_dir,
                extract=True,
            )
            self.ppl_model = TFGPT2LMHeadModel.from_pretrained(path)
            self.ppl_tokenizer = _XLNetTokenizer.from_pretrained(path)
        try:
            self.ppl_model.to(default_device)
        except Exception:
            self.ppl_model.to(torch.device("cpu"))
        self.ppl_model.eval()
        self.max_length = self.ppl_model.config.n_positions
        self.stride = 512

    def calculate(self, results: Sequence[GoalFunctionResult]) -> dict:
        """ """
        self.results = results
        self.original_candidates_ppl = []
        self.successful_candidates_ppl = []

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text.text.lower()
                )
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text.text.lower()
                )

        ppl_orig = self.calc_ppl(self.original_candidates)
        ppl_attack = self.calc_ppl(self.successful_candidates)

        self.all_metrics["avg_original_perplexity"] = round(ppl_orig[0], 2)
        self.all_metrics["original_perplexity_list"] = ppl_orig[1]

        self.all_metrics["avg_attack_perplexity"] = round(ppl_attack[0], 2)
        self.all_metrics["attack_perplexity_list"] = ppl_attack[1]

        return self.all_metrics

    def calc_ppl(self, texts: Sequence[str]) -> tuple:
        """
        计算 perplexity
        """
        ppl_vals = []

        with torch.no_grad():
            for text in texts:
                eval_loss = []
                input_ids = torch.tensor(
                    self.ppl_tokenizer.encode(text, add_special_tokens=True)
                ).unsqueeze(0)
                # Strided perplexity calculation from huggingface.co/transformers/perplexity.html
                for i in range(0, input_ids.size(1), self.stride):
                    begin_loc = max(i + self.stride - self.max_length, 0)
                    end_loc = min(i + self.stride, input_ids.size(1))
                    trg_len = end_loc - i
                    input_ids_t = input_ids[:, begin_loc:end_loc].to(
                        self.ppl_model.device
                    )
                    target_ids = input_ids_t.clone()
                    target_ids[:, :-trg_len] = -100

                    outputs = self.ppl_model(input_ids_t, labels=target_ids)
                    log_likelihood = outputs[0] * trg_len

                    eval_loss.append(log_likelihood)

                ppl_vals.append(
                    torch.exp(torch.stack(eval_loss).sum() / end_loc).item()
                )

        return sum(ppl_vals) / len(ppl_vals), ppl_vals


# add spicel process
class _XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        return text
