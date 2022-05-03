# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-27
@LastEditTime: 2021-09-07
"""

from typing import Union, NoReturn, List

import lru

from .base import Constraint
from ..transformations import (
    Transformation,
    transformation_consists_of_word_substitutes,
)
from ..strings import (  # noqa: F401
    words_from_text,
    LANGUAGE,
    normalize_language,
    UNIVERSAL_POSTAG,
    normalize_pos_tag,
)
from ..strings_cn import jieba_tag, stanza_tag as stanza_tag_cn
from ..strings_en import flair_tag, nltk_tag, stanza_tag as stanza_tag_en
from ..attacked_text import AttackedText


__all__ = [
    "PartOfSpeech",
]


class PartOfSpeech(Constraint):
    """词性（part-of-speech, POS）限制

    Uses jieba by default for Chinese,
    the NLTK universal part-of-speech tagger by default for English

    An implementation
    of `<https://arxiv.org/abs/1907.11932>`_ adapted from
    `<https://github.com/jind11/TextFooler>`_.

    POS taggers from Flair `<https://github.com/flairNLP/flair>`_ and
    Stanza `<https://github.com/stanfordnlp/stanza>`_ are also available
    """

    def __init__(
        self,
        language: str = "zh",
        tagger: str = "jieba",
        allow_verb_noun_swap: bool = True,
        compare_against_original: bool = True,
    ) -> NoReturn:
        """
        @description: POS constraint
        @param {
            language:
            tagger: Name of the tagger to use (available choices: "jieba", "nltk", "flair", "stanza").
            allow_verb_noun_swap: If `True`, allow verbs to be swapped with nouns and vice versa.
            compare_against_original: If `True`, compare against the original text.
                Otherwise, compare against the most recent text.
        }
        @return: None
        """
        super().__init__(compare_against_original)
        self.tagger = tagger.lower()
        self.allow_verb_noun_swap = allow_verb_noun_swap
        self.language = normalize_language(language)

        self._pos_tag_cache = lru.LRU(2**14)

    def clear_cache(self) -> NoReturn:
        self._pos_tag_cache.clear()

    def _can_replace_pos(
        self, pos_a: Union[str, UNIVERSAL_POSTAG], pos_b: Union[str, UNIVERSAL_POSTAG]
    ) -> bool:
        """ """
        _pos_a = normalize_pos_tag(pos_a).name
        _pos_b = normalize_pos_tag(pos_b).name
        return (_pos_a == _pos_b) or (
            self.allow_verb_noun_swap and set([_pos_a, _pos_b]) <= set(["NOUN", "VERB"])
        )

    def _get_pos(
        self, before_ctx: List[str], word: str, after_ctx: List[str]
    ) -> UNIVERSAL_POSTAG:
        context_words = before_ctx + [word] + after_ctx
        context_key = " ".join(context_words)
        if context_key in self._pos_tag_cache:
            word_list, pos_list = self._pos_tag_cache[context_key]
        else:
            if self.tagger == "jieba":
                tmp = jieba_tag(context_key)
            elif self.tagger == "flair":
                tmp = flair_tag(context_key)
            elif self.tagger == "nltk":
                if self.language == LANGUAGE.CHINESE:
                    raise ValueError("nltk 不能用于中文的 POS tagging")
                tmp = nltk_tag(context_key)
            elif self.tagger == "stanza":
                if self.language == LANGUAGE.CHINESE:
                    tagger = stanza_tag_cn
                elif self.language == LANGUAGE.ENGLISH:
                    tagger = stanza_tag_en
                tmp = tagger(context_key)
            word_list, pos_list = tmp["words"], tmp["tags"]

            self._pos_tag_cache[context_key] = (word_list, pos_list)

        # idx of `word` in `context_words`
        assert word in word_list, "POS list not matched with original word list."
        word_idx = word_list.index(word)
        return pos_list[word_idx]

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        for i in indices:
            reference_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]
            before_ctx = reference_text.words[max(i - 4, 0) : i]
            after_ctx = reference_text.words[
                i + 1 : min(i + 4, len(reference_text.words))
            ]
            ref_pos = self._get_pos(before_ctx, reference_word, after_ctx)
            replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
            if not self._can_replace_pos(ref_pos, replace_pos):
                return False

        return True

    def check_compatibility(self, transformation: Transformation) -> bool:
        return transformation_consists_of_word_substitutes(transformation)

    def extra_repr_keys(self) -> List[str]:
        return [
            "tagger",
            "tagset",
            "allow_verb_noun_swap",
        ] + super().extra_repr_keys()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pos_tag_cache"] = self._pos_tag_cache.get_size()
        return state

    def __setstate__(self, state) -> NoReturn:
        self.__dict__ = state
        self._pos_tag_cache = lru.LRU(state["_pos_tag_cache"])
