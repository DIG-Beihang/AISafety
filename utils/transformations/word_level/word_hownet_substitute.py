# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-21
"""

import random
from typing import NoReturn, Optional, List

from ..base import WordSubstitute
from ...hownet import HowNetDict
from ...assets import fetch
from ...strings import (
    normalize_pos_tag,
    UNIVERSAL_POSTAG,
    normalize_language,
    LANGUAGE,
)


__all__ = [
    "WordHowNetSubstitute",
]


class WordHowNetSubstitute(WordSubstitute):
    """ """

    __VALID_POS = {
        UNIVERSAL_POSTAG.NOUN: "noun",
        UNIVERSAL_POSTAG.VERB: "verb",
        UNIVERSAL_POSTAG.ADJ: "adj",
        UNIVERSAL_POSTAG.ADV: "adv",
    }
    __name__ = "WordHowNetSubstitute"

    def __init__(
        self, language: str, max_candidates: int = -1, verbose: int = 0
    ) -> NoReturn:
        """ """
        super().__init__()
        self._language = normalize_language(language)
        if self._language == LANGUAGE.ENGLISH:
            self._hnd = fetch("hownet", language="en")
        elif self._language == LANGUAGE.CHINESE:
            self._hnd = fetch("hownet", language="zh")
        else:
            raise ValueError(f"暂不支持{language}")
        self.max_candidates = max_candidates
        self.verbose = verbose

    @property
    def deterministic(self) -> bool:
        return False

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """ """
        _pos_tag = normalize_pos_tag(pos_tag)
        if _pos_tag is None or word not in self._hnd:
            return [word]
        elif (
            _pos_tag not in self.__VALID_POS
            or self.__VALID_POS[_pos_tag] not in self._hnd[word]
        ):
            return [word]

        res = self._hnd[word][self.__VALID_POS[_pos_tag]]
        random.shuffle(res)
        _num = num or self.max_candidates
        if _num > 0:
            res = res[:_num]
        return res

    @property
    def valid_pos(self) -> dict:
        return self.__VALID_POS


class _DeprecatedWordHowNetSubstitute(WordSubstitute):
    """ """

    __VALID_POS = {
        UNIVERSAL_POSTAG.NOUN: "noun",
        UNIVERSAL_POSTAG.VERB: "verb",
        UNIVERSAL_POSTAG.ADJ: "adj",
        UNIVERSAL_POSTAG.ADV: "adv",
    }
    __name__ = "_DeprecatedWordHowNetSubstitute"

    def __init__(self) -> NoReturn:
        super().__init__()
        self._hnd = HowNetDict()
        self.en_word_list = self._hnd.get_en_words()

    @property
    def deterministic(self) -> bool:
        return False

    def _get_candidates(
        self, word: str, pos_tag: Optional[str] = None, num: Optional[int] = None
    ) -> List[str]:
        """ """
        _pos_tag = normalize_pos_tag(pos_tag)
        if _pos_tag is None:
            return [word]

        word_candidates = []

        if _pos_tag not in self.__VALID_POS:
            raise ValueError(f"暂不支持POS {pos_tag}")

        # get sememes
        word_sememes = self._hnd.get_sememes_by_word(
            word, structured=False, language="en", merge=False
        )
        word_sememe_sets = [t["sememes"] for t in word_sememes]
        if len(word_sememes) == 0:
            return [word]

        pos_set = set(self.__VALID_POS.values())
        _max_num = num or float("inf")

        # find candidates
        shuffled_range = list(range(len(self.en_word_list)))
        random.shuffle(shuffled_range)
        for idx in shuffled_range:
            wd = self.en_word_list[idx]
            if wd is word or len(wd.split(" ")) != 1:
                continue
            # pos
            word_pos = set()
            word_pos.add(self.__VALID_POS[_pos_tag])
            result_list2 = self._hnd.get(wd)
            wd_pos = set()
            for a in result_list2:
                if type(a) != dict:
                    continue
                wd_pos.add(a["en_grammar"])
            all_pos = wd_pos & word_pos & pos_set
            if len(all_pos) == 0:
                continue

            # sememe
            wd_sememes = self._hnd.get_sememes_by_word(
                wd, structured=False, language="en", merge=False
            )
            wd_sememe_sets = [t["sememes"] for t in wd_sememes]
            if len(wd_sememes) == 0:
                continue
            can_be_sub = False
            for s1 in word_sememe_sets:
                for s2 in wd_sememe_sets:
                    if s1 == s2:
                        can_be_sub = True
                        break
            if can_be_sub:
                word_candidates.append(wd)

            if len(word_candidates) >= _max_num:
                break

        return word_candidates

    def _get_candidates_textattack(
        self, word: str, pos_tag: Optional[str] = None, num: Optional[int] = None
    ) -> List[str]:
        """
        源自TextAttack的方法
        """
        raise NotImplementedError

    @property
    def valid_pos(self) -> dict:
        return self.__VALID_POS
