# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-10-29
@LastEditTime: 2021-10-29
"""

import re
from typing import Any, List, NoReturn, Sequence, Tuple, Optional

import torch

from ..base import WordSubstitute
from ...strings import normalize_language, LANGUAGE, isChinese
from ...word_embeddings import WordEmbedding
from ...strings_en import tokenize
from ...attacked_text import AttackedText
from ....Models.base import NLPVictimModel


__all__ = [
    "WordEmbeddingGradientSignSubstitute",
]


class WordEmbeddingGradientSignSubstitute(WordSubstitute):
    """Uses the model's gradient and word embedding to suggest replacements for a given word.

    Reference
    ---------
    Papernot N, McDaniel P, Swami A, et al. Crafting adversarial input sequences for recurrent neural networks[C]//MILCOM 2016-2016 IEEE Military Communications Conference. IEEE, 2016: 49-54.
    """

    __name__ = "WordEmbeddingGradientSignSubstitute"

    def __init__(
        self,
        language: str,
        embedding: WordEmbedding,
        model_wrapper: NLPVictimModel,
        top_n: int = 5,
        verbose: int = 0,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Arguments:
            model: The model to attack. Model must have a
                `word_embeddings` matrix and `convert_id_to_word` function.
            top_n: the number of top words to return at each index
        """
        self.language = normalize_language(language)
        self.embedding = embedding
        self.verbose = verbose
        # Unwrap model wrappers. Need raw model for gradient.
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        # Make sure we know how to compute the gradient for this model.
        # validate_model_gradient_word_swap_compatibility(self.model)
        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id:
            raise ValueError(
                "Tokenizer needs to have `pad_token_id` for gradient-based word swap"
            )

        self.top_n = top_n
        self.is_black_box = False

    def _get_candidates(self, word: str, num: int = None) -> List[str]:
        """Based on nearest neighbors selected word embeddings."""
        try:
            nnids = self.embedding.nearest_neighbours(word, num or 2 * self.top_n)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.embedding.index2word(nbr_id)
                if self.language == LANGUAGE.CHINESE:
                    if not isChinese(nbr_word, strict=True):
                        continue
                elif self.language == LANGUAGE.ENGLISH:
                    if len(re.findall("[a-zA-Z]", nbr_word)) == 0 or (
                        len(tokenize(nbr_word)) != 1
                    ):
                        # Do not consider words that are solely letters or punctuation.
                        continue
                candidate_words.append(_recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []
        if self.verbose >= 2:
            print(
                f"using {self.__name__}, one get candidate_words\n{candidate_words}\ntransformed from `{word}`"
            )

    def _get_replacement_words_by_grad_sign(
        self,
        attacked_text: AttackedText,
        indices_to_replace: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.
        """
        _max_num = max_num or self.top_n
        emb_layer = self.model.get_input_embeddings()

        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"])
        text_ids = grad_output["ids"]
        if text_ids.ndim > 1:
            text_ids = text_ids.squeeze()

        candidates = []

        for j, word_idx in enumerate(indices_to_replace):
            # Make sure the word is in bounds.
            if word_idx >= len(emb_grad):
                continue
            word_emb = emb_layer(
                torch.as_tensor(
                    self.tokenizer.convert_tokens_to_ids(attacked_text.words[word_idx]),
                    dtype=torch.long,
                )
            )
            new_words = self._get_candidates(attacked_text.words[word_idx])
            new_words_emb = emb_layer(
                torch.as_tensor(
                    self.tokenizer.convert_tokens_to_ids(new_words),
                    dtype=torch.long,
                )
            )
            sign_emb = (word_emb - new_words_emb).sign()
            # Get the grad w.r.t the one-hot index of the word.
            sign_grads = emb_grad[word_idx].sign()
            # print(f"sign_emb.shape = {sign_emb.shape}, sign_grads.shape = {sign_grads.shape}")
            # print(f"word = {attacked_text.words[word_idx]}, new_words = {new_words}")

            emb_grad_sign = abs(sign_emb - sign_grads).abs().sum(-1)

            for idx in emb_grad_sign.argsort()[: self.top_n]:
                candidates.append((new_words[idx], word_idx))

        if self.verbose >= 2:
            print(f"using transformation {self.__name__}")
            print(
                f"from original text {attacked_text.text}, one get the following candidates"
            )
            print(candidates)

        return candidates

    def _get_transformations(
        self,
        attacked_text: AttackedText,
        indices_to_replace: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those indices.
        """
        transformations = []
        for word, idx in self._get_replacement_words_by_grad_sign(
            attacked_text,
            indices_to_replace,
            max_num,
        ):
            transformations.append(attacked_text.replace_word_at_index(idx, word))
        return transformations

    def extra_repr_keys(self) -> List[str]:
        return [
            "top_n",
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
