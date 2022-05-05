# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-11-10

Word Merge by BERT-Masked LM.
"""

import os
from typing import Any, List, NoReturn, Sequence, Optional, Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BatchEncoding

from ..base import Transformation
from ...strings import (  # noqa: F401
    normalize_language,
    LANGUAGE,
    normalize_pos_tag,
    UNIVERSAL_POSTAG,
    check_if_subword,
    check_if_punctuations,
    strip_BPE_artifacts,
    isChinese,
)
from ...strings_en import tokenize
from ...misc import nlp_cache_dir, default_device
from ...attacked_text import AttackedText


__all__ = [
    "WordMaskedLMMerge",
]


class WordMaskedLMMerge(Transformation):
    """Generate potential merge of adjacent using a masked language model.

    Based off of:
        CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020)
        https://arxiv.org/abs/2009.07502
    """

    __name__ = "WordMaskedLMMerge"

    def __init__(
        self,
        language: str,
        masked_lm_or_path: Union[str, AutoModelForMaskedLM] = "bert-base-uncased",
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        window_size: Union[int, float] = float("inf"),
        max_candidates: int = 50,
        min_confidence: float = 5e-4,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Args:
            masked_lm_or_path:
                Either the name of pretrained masked language model from `transformers` model hub
                or the actual model. Default is `bert-base-uncased`.
            tokenizer:
                The tokenizer of the corresponding model.
                If you passed in name of a pretrained model for `masked_language_model`,
                you can skip this argument as the correct tokenizer can be infered from the name.
                However, if you're passing the actual model, you must provide a tokenizer.
            max_length:
                The max sequence length the masked language model is designed to work with. Default is 512.
            window_size:
                The number of surrounding words to include when making top word prediction.
                For each position to merge, we take `window_size // 2` words to the left
                and `window_size // 2` words to the right and pass the text within the window
                to the masked language model. Default is `float("inf")`, which is equivalent to using the whole text.
            max_candidates:
                Maximum number of candidates to consider as replacements for each word.
                Replacements are ranked by model's confidence.
            min_confidence:
                Minimum confidence threshold each replacement word must pass.
        """
        super().__init__()
        self._language = normalize_language(language)
        self.max_length = max_length
        self.window_size = window_size
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size

        if isinstance(masked_lm_or_path, str):
            if os.path.exists(masked_lm_or_path):
                _load_path = masked_lm_or_path
            elif os.path.exists(os.path.join(nlp_cache_dir, masked_lm_or_path)):
                _load_path = os.path.join(nlp_cache_dir, masked_lm_or_path)
            else:
                _load_path = None
                # raise ValueError(f"本地模型不存在，暂不支持在线从HuggingFace Hub载入模型{masked_lm_or_path}")
                self._language_model = AutoModelForMaskedLM.from_pretrained(
                    masked_lm_or_path
                )
                self._lm_tokenizer = AutoTokenizer.from_pretrained(
                    masked_lm_or_path, use_fast=True
                )
            if _load_path:
                self._language_model = AutoModelForMaskedLM.from_pretrained(_load_path)
                self._lm_tokenizer = AutoTokenizer.from_pretrained(
                    _load_path, use_fast=True
                )
        else:
            self._language_model = masked_lm_or_path
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        try:
            self._language_model.to(default_device)
        except Exception:
            self._language_model.to(torch.device("cpu"))
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__

    def _encode_text(self, text: str) -> BatchEncoding:
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.to(self._language_model.device)
        # try:
        #     return encoding.to(default_device)
        # except Exception:
        #     return encoding.to(torch.device("cpu"))

    def _get_candidates(
        self, current_text: AttackedText, indices_to_modify: Sequence[int]
    ) -> List[List[str]]:
        """Get replacement words for the word we want to replace using BAE method.

        Args:
            current_text: Text we want to get replacements for.
            indices_to_modify: indices of words to replace
        """
        masked_texts = []
        for index in indices_to_modify:
            temp_text = current_text.replace_word_at_index(
                index, self._lm_tokenizer.mask_token
            )
            temp_text = temp_text.delete_word_at_index(index + 1)
            # Obtain window
            temp_text = temp_text.text_window_around_index(index, self.window_size)
            masked_texts.append(temp_text)

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = [
                inputs["input_ids"][i].tolist() for i in range(len(inputs["input_ids"]))
            ]
            with torch.no_grad():
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    replacement_words.append([])
                    continue

                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                top_words = []
                for _id in ranked_indices:
                    _id = _id.item()
                    word = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    if check_if_subword(
                        word,
                        self._language_model.config.model_type,
                        (masked_index == 1),
                    ):
                        word = strip_BPE_artifacts(
                            word, self._language_model.config.model_type
                        )
                    if (
                        mask_token_probs[_id] >= self.min_confidence
                        and len(tokenize(word)) == 1
                        and not check_if_punctuations(word)
                    ):
                        top_words.append(word)

                    if (
                        len(top_words) >= self.max_candidates
                        or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_words.append(top_words)

            i += self.batch_size

        return replacement_words

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        transformed_texts = []
        indices_to_modify = list(indices_to_modify)
        # find indices that are suitable to merge
        token_tags = [
            current_text.pos_of_word_index(i) for i in range(current_text.num_words)
        ]
        merge_indices = find_merge_index(token_tags)
        merged_words = self._get_candidates(current_text, merge_indices)
        transformed_texts = []
        for i in range(len(merged_words)):
            index_to_modify = merge_indices[i]
            word_at_index = current_text.words[index_to_modify]
            for word in merged_words[i]:
                word = word.strip("Ġ")
                if word != word_at_index:
                    temp_text = current_text.delete_word_at_index(index_to_modify + 1)
                    transformed_texts.append(
                        temp_text.replace_word_at_index(index_to_modify, word)
                    )

        return transformed_texts

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["masked_lm_name", "max_length", "max_candidates", "min_confidence"]


_merge_map = {
    UNIVERSAL_POSTAG.NOUN: [
        UNIVERSAL_POSTAG.NOUN,
    ],
    UNIVERSAL_POSTAG.ADJ: [
        UNIVERSAL_POSTAG.NOUN,
        UNIVERSAL_POSTAG.NUM,
        UNIVERSAL_POSTAG.ADJ,
        UNIVERSAL_POSTAG.ADV,
    ],
    UNIVERSAL_POSTAG.ADV: [
        UNIVERSAL_POSTAG.ADJ,
        UNIVERSAL_POSTAG.VERB,
    ],
    UNIVERSAL_POSTAG.VERB: [
        UNIVERSAL_POSTAG.ADV,
        UNIVERSAL_POSTAG.VERB,
        UNIVERSAL_POSTAG.NOUN,
        UNIVERSAL_POSTAG.ADJ,
    ],
    UNIVERSAL_POSTAG.DET: [
        UNIVERSAL_POSTAG.NOUN,
        UNIVERSAL_POSTAG.ADJ,
    ],
    UNIVERSAL_POSTAG.PRON: [
        UNIVERSAL_POSTAG.NOUN,
        UNIVERSAL_POSTAG.ADJ,
    ],
    UNIVERSAL_POSTAG.NUM: [
        UNIVERSAL_POSTAG.NUM,
        UNIVERSAL_POSTAG.NOUN,
    ],
}


def find_merge_index(
    token_tags: Sequence[Union[str, UNIVERSAL_POSTAG]],
    indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """ """
    merge_indices = []
    if indices is None:
        indices = range(len(token_tags) - 1)
    for i in indices:
        cur_tag = normalize_pos_tag(token_tags[i])
        next_tag = normalize_pos_tag(token_tags[i + 1])
        if cur_tag in _merge_map and next_tag in _merge_map[cur_tag]:
            merge_indices.append(i)
    return merge_indices
