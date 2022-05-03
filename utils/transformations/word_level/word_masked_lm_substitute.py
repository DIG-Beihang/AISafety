# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-11-02

Word Substitute by BERT-Masked LM.
"""

import os
import itertools
from typing import Any, List, NoReturn, Sequence, Optional, Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BatchEncoding

from ..base import WordSubstitute
from ...strings import (  # noqa: F401
    normalize_language,
    LANGUAGE,
    check_if_subword,
    check_if_punctuations,
    strip_BPE_artifacts,
    isChinese,
)
from ...strings_en import tokenize
from ...misc import nlp_cache_dir, default_device
from ...attacked_text import AttackedText


__all__ = [
    "WordMaskedLMSubstitute",
]


class WordMaskedLMSubstitute(WordSubstitute):
    """Generate potential replacements for a word using a masked language model.

    Based off of following papers
        - "Robustness to Modification with Shared Words in Paraphrase Identification" (Shi et al., 2019) https://arxiv.org/abs/1909.02560
        - "BAE: BERT-based Adversarial Examples for Text Classification" (Garg et al., 2020) https://arxiv.org/abs/2004.01970
        - "BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al, 2020) https://arxiv.org/abs/2004.09984
        - "CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020): https://arxiv.org/abs/2009.07502

    BAE and CLARE simply masks the word we want to replace and selects replacements predicted by the masked language model.

    BERT-Attack instead performs replacement on token level. For words that are consisted of two or more sub-word tokens,
        it takes the top-K replacements for seach sub-word token and produces all possible combinations of the top replacments.
        Then, it selects the top-K combinations based on their perplexity calculated using the masked language model.

    Choose which method to use by specifying "bae" or "bert-attack" for `method` argument.
    """

    __name__ = "WordMaskedLMSubstitute"

    def __init__(
        self,
        language: str,
        method: str = "bae",
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
            method:
                the name of replacement method (e.g. "bae", "bert-attack")
            masked_lm_or_path:
                Either the name or local path of pretrained masked language model from `transformers` model hub
                or the actual model. Default is `bert-base-uncased`.
            tokenizer:
                The tokenizer of the corresponding model. If you passed in name of a pretrained model for `masked_lm_or_path`,
                you can skip this argument as the correct tokenizer can be infered from the name.
                However, if you're passing the actual model, you must provide a tokenizer.
            max_length:
                the max sequence length the masked language model is designed to work with. Default is 512.
            window_size:
                The number of surrounding words to include when making top word prediction.
                For each word to swap, we take `window_size // 2` words to the left and `window_size // 2` words to the right and pass the text within the window
                to the masked language model. Default is `float("inf")`, which is equivalent to using the whole text.
            max_candidates:
                maximum number of candidates to consider as replacements for each word. Replacements are ranked by model's confidence.
            min_confidence:
                minimum confidence threshold each replacement word must pass.
            batch_size:
                Size of batch for "bae" replacement method.
        """
        super().__init__()
        self._language = normalize_language(language)
        self.method = method
        self.max_length = max_length
        self.window_size = window_size
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.verbose = kwargs.get("verbose", 0)

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

    def _bae_replacement_words(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[List[str]]:
        """Get replacement words for the word we want to replace using BAE method."""
        masked_texts = []
        for index in indices_to_modify:
            masked_text = current_text.replace_word_at_index(
                index, self._lm_tokenizer.mask_token
            )
            # Obtain window
            masked_text = masked_text.text_window_around_index(index, self.window_size)
            masked_texts.append(masked_text)

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = inputs["input_ids"].tolist()
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

    def _bert_attack_replacement_words(
        self,
        current_text: AttackedText,
        index: int,
        id_preds: torch.Tensor,
        masked_lm_logits: torch.Tensor,
    ) -> list:
        """Get replacement words for the word we want to replace using BERT-Attack method.

        Args:
            current_text: Text we want to get replacements for.
            index: index of word we want to replace
            id_preds: N x K tensor of top-K ids for each token-position predicted by the masked language model.
                N is equivalent to `self.max_length`.
            masked_lm_logits: N x V tensor of the raw logits outputted by the masked language model.
                N is equivlaent to `self.max_length` and V is dictionary size of masked language model.
        """
        # We need to find which BPE tokens belong to the word we want to replace
        masked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        current_inputs = self._encode_text(masked_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]
        word_tokens = self._lm_tokenizer.encode(
            current_text.words[index], add_special_tokens=False
        )

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        # List of indices of tokens that are part of the target word
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(word_tokens), self.max_length))
        )

        if not len(target_ids_pos):
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = id_preds[target_ids_pos[0]].tolist()
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if len(tokenize(token)) == 1 and not check_if_subword(
                    token, self._language_model.config.model_type, index == 0
                ):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [id_preds[i] for i in target_ids_pos]
            products = itertools.product(*top_preds)
            combination_results = []
            # Original BERT-Attack implement uses cross-entropy loss
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)
            word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)
            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    word_tensor[i] = bpe_tokens[i]

                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()
                word = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
                ).replace("##", "")
                # if is_one_word(word):
                if len(tokenize(word)) == 1:
                    combination_results.append((word, perplexity))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]
            return top_replacements

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        indices_to_modify = list(indices_to_modify)
        if self.method == "bert-attack":
            current_inputs = self._encode_text(current_text.text)
            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

            transformed_texts = []

            for i in indices_to_modify:
                word_at_index = current_text.words[i]
                replacement_words = self._bert_attack_replacement_words(
                    current_text,
                    i,
                    id_preds=id_preds,
                    masked_lm_logits=masked_lm_logits,
                )

                for r in replacement_words:
                    r = r.strip("Ġ")
                    if r != word_at_index:
                        transformed_texts.append(
                            current_text.replace_word_at_index(i, r)
                        )

            return transformed_texts

        elif self.method == "bae":
            replacement_words = self._bae_replacement_words(
                current_text, indices_to_modify, max_num
            )
            transformed_texts = []
            for i in range(len(replacement_words)):
                index_to_modify = indices_to_modify[i]
                word_at_index = current_text.words[index_to_modify]
                for word in replacement_words[i]:
                    word = word.strip("Ġ")
                    if word != word_at_index and len(tokenize(word)) == 1:
                        transformed_texts.append(
                            current_text.replace_word_at_index(index_to_modify, word)
                        )
            return transformed_texts
        else:
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")

    def _get_candidates(
        self, word: str, pos_tag: str = None, num: int = None
    ) -> List[str]:
        """ """
        pass

    def extra_repr_keys(self) -> List[str]:
        return [
            "method",
            "masked_lm_name",
            "max_length",
            "max_candidates",
            "min_confidence",
        ]


def _recover_word_case(word: str, reference_word: str) -> str:
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
