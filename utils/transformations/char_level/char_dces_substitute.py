# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-24
@LastEditTime: 2022-04-17

源自OpenAttack的DCESSubstitute

"""

import random
from typing import NoReturn, List, Any, Optional

import numpy as np

from ..base import CharSubstitute
from ...assets import fetch
from ...misc import DEFAULTS


__all__ = [
    "CharacterDCESSubstitute",
]


class CharacterDCESSubstitute(CharSubstitute):
    """ """

    __name__ = "CharacterDCESSubstitute"

    def __init__(
        self, threshold: float, random_one: bool = False, **kwargs: Any
    ) -> NoReturn:
        """ """
        super().__init__(**kwargs)
        self.threshold = threshold
        dces_dict = fetch("dces")
        self.descs = dces_dict["descs"]
        self.neigh = dces_dict["neigh"]
        self.random_one = random_one

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """ """
        candidate_words = []

        if self.random_one:
            i = DEFAULTS.RNG.integers(0, len(word))
            repl_letters = self._apply_dces(word[i], self.threshold)
            if len(repl_letters) > 0:
                repl_letter = random.choice(repl_letters)
                candidate_word = word[:i] + repl_letter + word[i + 1 :]
                candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                for repl_letter in self._apply_dces(word[i], self.threshold):
                    candidate_word = word[:i] + repl_letter + word[i + 1 :]
                    candidate_words.append(candidate_word)
        if num:
            candidate_words = candidate_words[:num]

        return candidate_words

    def _apply_dces(self, char: str, threshold: float) -> List[str]:
        """ """
        c = get_hex_string(char)

        if c in self.descs:
            description = self.descs[c]["description"]
        else:
            return []

        tokens = description.split(" ")
        case = "unknown"
        identifiers = []

        for token in tokens:
            if len(token) == 1:
                identifiers.append(token)
            elif token == "SMALL":
                case = "SMALL"
            elif token == "CAPITAL":
                case = "CAPITAL"

        matches = []
        match_ids = []
        for i in identifiers:
            for idx, val in self.descs.items():
                desc_toks = val["description"].split(" ")
                if (
                    i in desc_toks
                    and not np.any(np.in1d(desc_toks, _disallowed))
                    and not np.any(np.in1d(idx, _disallowed_codes))
                    and not int(idx, 16) > 30000
                ):

                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[
                        (desc_toks == "SMALL") | (desc_toks == "CAPITAL")
                    ]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = "unknown"

                    if case == "unknown" or case == case_descriptor:
                        match_ids.append(idx)
                        matches.append(val["vec"])

        if len(matches) == 0:
            return []

        match_vecs = np.stack(matches)
        Y = match_vecs

        self.neigh.fit(Y)

        X = self.descs[c]["vec"].reshape(1, -1)

        if Y.shape[0] > threshold:
            dists, idxs = self.neigh.kneighbors(X, threshold, return_distance=True)
        else:
            dists, idxs = self.neigh.kneighbors(X, Y.shape[0], return_distance=True)
        probs = dists.flatten()

        charcodes = [match_ids[idx] for idx in idxs.flatten()]

        chars = []
        for idx, charcode in enumerate(charcodes):
            if probs[idx] < threshold:
                chars.append(chr(int(charcode, 16)))
        return chars

    @property
    def deterministic(self) -> bool:
        return not self.random_one

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "threshold",
            "random_one",
        ]


_disallowed = [
    "TAG",
    "MALAYALAM",
    "BAMUM",
    "HIRAGANA",
    "RUNIC",
    "TAI",
    "SUNDANESE",
    "BATAK",
    "LEPCHA",
    "CHAM",
    "TELUGU",
    "DEVANGARAI",
    "BUGINESE",
    "MYANMAR",
    "LINEAR",
    "SYLOTI",
    "PHAGS-PA",
    "CHEROKEE",
    "CANADIAN",
    "YI",
    "LYCIAN",
    "HANGUL",
    "KATAKANA",
    "JAVANESE",
    "ARABIC",
    "KANNADA",
    "BUHID",
    "TAGBANWA",
    "DESERET",
    "REJANG",
    "BOPOMOFO",
    "PERMIC",
    "OSAGE",
    "TAGALOG",
    "MEETEI",
    "CARIAN",
    "UGARITIC",
    "ORIYA",
    "ELBASAN",
    "CYPRIOT",
    "HANUNOO",
    "GUJARATI",
    "LYDIAN",
    "MONGOLIAN",
    "AVESTAN",
    "MEROITIC",
    "KHAROSHTHI",
    "HUNGARIAN",
    "KHUDAWADI",
    "ETHIOPIC",
    "PERSIAN",
    "OSMANYA",
    "ELBASAN",
    "TIBETAN",
    "BENGALI",
    "TURKIC",
    "THROWING",
    "HANIFI",
    "BRAHMI",
    "KAITHI",
    "LIMBU",
    "LAO",
    "CHAKMA",
    "DEVANAGARI",
    "ITALIC",
    "CJK",
    "MEDEFAIDRIN",
    "DIAMOND",
    "SAURASHTRA",
    "ADLAM",
    "DUPLOYAN",
]
_disallowed_codes = [
    "1F1A4",
    "A7AF",
]


def get_hex_string(ch: str) -> str:
    return "{:04x}".format(ord(ch)).upper()
