# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-13
"""

import random
from typing import NoReturn, Optional, List

from .word_hownet_substitute import WordHowNetSubstitute
from ..base import WordSubstitute
from ...hownet import HowNetDict
from ...strings import normalize_pos_tag, UNIVERSAL_POSTAG


__all__ = [
    "ChineseHowNetSubstitute",
]


class ChineseHowNetSubstitute(WordHowNetSubstitute):
    """ """

    __name__ = "ChineseHowNetSubstitute"

    def __init__(self, max_candidates: int = -1, verbose: int = 0) -> NoReturn:
        """ """
        super().__init__(language="zh", max_candidates=max_candidates, verbose=verbose)


class _DeprecatedChineseHowNetSubstitute(WordSubstitute):
    """ """

    __VALID_POS = {
        UNIVERSAL_POSTAG.NOUN: "noun",
        UNIVERSAL_POSTAG.VERB: "verb",
        UNIVERSAL_POSTAG.ADJ: "adj",
        UNIVERSAL_POSTAG.ADV: "adv",
    }
    __name__ = "_DeprecatedChineseHowNetSubstitute"

    def __init__(self) -> NoReturn:
        super().__init__()
        self._hnd = HowNetDict()
        self.zh_word_list = self._hnd.get_zh_words()

    @property
    def deterministic(self) -> bool:
        return False

    def _get_syn(self, word: str, pos_tag: str, num: Optional[int] = None) -> List[str]:
        """ """
        word_sememes = self._hnd.get_sememes_by_word(
            word, structured=False, language="zh", merge=False
        )
        word_sememe_sets = [t["sememes"] for t in word_sememes]
        raw_res = []
        for item in self._hnd.get(word):
            if item["ch_grammar"] != pos_tag:
                continue
            raw_res.extend([d["text"] for d in item["syn"]])
        res = []
        for wd in set(raw_res):
            wd_sememes = self._hnd.get_sememes_by_word(
                wd, structured=False, language="zh", merge=False
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
                res.append(wd)
        random.shuffle(res)
        if num:
            res = res[:num]
        return res

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
        require_syn: bool = False,
    ) -> List[str]:
        """ """
        _pos_tag = normalize_pos_tag(pos_tag)
        if _pos_tag is None:
            return [word]

        if require_syn:
            return self._get_syn(word, pos_tag, num)

        word_candidates = []

        if _pos_tag not in self.__VALID_POS:
            raise ValueError(f"暂不支持POS {pos_tag}")

        # get sememes
        word_sememes = self._hnd.get_sememes_by_word(
            word, structured=False, language="zh", merge=False
        )
        word_sememe_sets = [t["sememes"] for t in word_sememes]
        if len(word_sememes) == 0:
            return [word]

        pos_set = set(self.__VALID_POS.values())
        _max_num = num or float("inf")

        # find candidates
        shuffled_range = list(range(len(self.zh_word_list)))
        random.shuffle(shuffled_range)
        for idx in shuffled_range:
            wd = self.zh_word_list[idx]
            if wd is word or len(wd.split(" ")) != 1:
                continue
            # pos
            word_pos = set()
            word_pos.add(self.__VALID_POS[_pos_tag])
            result_list2 = self._hnd.get(wd)
            wd_pos = set()
            for a in result_list2:
                if type(a) != dict or not a["ch_grammar"]:
                    continue
                wd_pos.add(a["ch_grammar"])
            all_pos = wd_pos & word_pos & pos_set
            if len(all_pos) == 0:
                continue

            # sememe
            wd_sememes = self._hnd.get_sememes_by_word(
                wd, structured=False, language="zh", merge=False
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

    def _get_candidates_openattack(
        self, word: str, pos_tag: Optional[str] = None, num: Optional[int] = None
    ) -> List[str]:
        """
        源自OpenAttack的方法
        """
        _pos_tag = normalize_pos_tag(pos_tag)
        if _pos_tag is None:
            return [word]

        word_candidates = []

        if _pos_tag not in self.__VALID_POS:
            raise ValueError(f"暂不支持POS {pos_tag}")

        # get sememes
        word_sememes = self._hnd.get_sememes_by_word(
            word, structured=False, language="zh", merge=False
        )
        word_sememe_sets = [t["sememes"] for t in word_sememes]
        if len(word_sememes) == 0:
            return [word]

        pos_set = set(self.__VALID_POS.values())

        # find candidates
        for wd in self.zh_word_list:
            if wd is word:
                continue

            # pos
            word_pos = set()
            word_pos.add(self.__VALID_POS[_pos_tag])
            result_list2 = self._hnd.get(wd)
            wd_pos = set()
            for a in result_list2:
                if type(a) != dict or not a["ch_grammar"]:
                    continue
                wd_pos.add(a["ch_grammar"])
            all_pos = wd_pos & word_pos & pos_set
            if len(all_pos) == 0:
                continue

            # sememe
            wd_sememes = self._hnd.get_sememes_by_word(
                wd, structured=False, language="zh", merge=False
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

        ret = []
        for wd in word_candidates:
            wdlist = wd.split(" ")
            if len(wdlist) == 1:
                ret.append(wd)
        random.shuffle(ret)
        if num is not None:
            ret = ret[:num]
        return ret

    @property
    def valid_pos(self) -> dict:
        return self.__VALID_POS
