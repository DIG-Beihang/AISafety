# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-13
@LastEditTime: 2022-04-17

源自TextAttack的WordSwapHomoglyphSwap，以及OpenAttack的ECESSubstitute

"""

import random
from typing import NoReturn, List, Any, Optional

from ..base import CharSubstitute
from ...misc import DEFAULTS


__all__ = [
    "CharacterHomoglyphSubstitute",
]


class CharacterHomoglyphSubstitute(CharSubstitute):
    """Transforms an input by replacing its words with visually similar words
    using homoglyph swaps."""

    __name__ = "CharacterHomoglyphSubstitute"

    def __init__(self, random_one: bool = False, **kwargs: Any) -> NoReturn:
        """ """
        super().__init__(**kwargs)
        self.homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        new_homos = {
            "a": "â",
            "b": "ḃ",
            "c": "ĉ",
            "d": "ḑ",
            "e": "ê",
            "f": "ḟ",
            "g": "ǵ",
            "h": "ĥ",
            "i": "î",
            "j": "ĵ",
            "k": "ǩ",
            "l": "ᶅ",
            "m": "ḿ",
            "n": "ň",
            "o": "ô",
            "p": "ṕ",
            "q": "ʠ",
            "r": "ř",
            "s": "ŝ",
            "t": "ẗ",
            "u": "ǔ",
            "v": "ṽ",
            "w": "ẘ",
            "x": "ẍ",
            "y": "ŷ",
            "z": "ẑ",
            "A": "Â",
            "B": "Ḃ",
            "C": "Ĉ",
            "D": "Ď",
            "E": "Ê",
            "F": "Ḟ",
            "G": "Ĝ",
            "H": "Ĥ",
            "I": "Î",
            "J": "Ĵ",
            "K": "Ǩ",
            "L": "Ĺ",
            "M": "Ḿ",
            "N": "Ň",
            "O": "Ô",
            "P": "Ṕ",
            "Q": "Q",
            "R": "Ř",
            "S": "Ŝ",
            "T": "Ť",
            "U": "Û",
            "V": "Ṽ",
            "W": "Ŵ",
            "X": "Ẍ",
            "Y": "Ŷ",
            "Z": "Ẑ",
        }
        self.homos = {
            k: [v, new_homos[k]] if k in new_homos else [v]
            for k, v in self.homos.items()
        }

        self.random_one = random_one

    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        candidate_words = []

        if self.random_one:
            i = DEFAULTS.RNG.integers(0, len(word))
            if word[i] in self.homos:
                repl_letter = random.choice(self.homos[word[i]])
                candidate_word = word[:i] + repl_letter + word[i + 1 :]
                candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                if word[i] in self.homos:
                    for repl_letter in self.homos[word[i]]:
                        candidate_word = word[:i] + repl_letter + word[i + 1 :]
                        candidate_words.append(candidate_word)
        if num:
            candidate_words = candidate_words[:num]

        return candidate_words

    @property
    def deterministic(self) -> bool:
        return not self.random_one

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "random_one",
        ]
