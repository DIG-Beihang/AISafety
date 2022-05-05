# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-21
@LastEditTime: 2021-10-29
"""

from .base import (
    Transformation,
    CharSubstitute,
    WordSubstitute,
    WordInsertion,
    WordDeletion,
    WordMerge,
    PhraseInsertion,
    CompositeTransformation,
    RandomCompositeTransformation,
    transformation_consists_of,
    transformation_consists_of_word_substitutes,
    transformation_consists_of_word_substitutes_and_deletions,
)
from .word_level.chinese_cilin_substitute import ChineseCiLinSubstitute
from .word_level.chinese_hownet_substitute import ChineseHowNetSubstitute
from .word_level.chinese_wordnet_substitute import ChineseWordNetSubstitute
from .word_level.wordnet_substitute import WordNetSubstitute
from .word_level.word_hownet_substitute import WordHowNetSubstitute
from .word_level.word_embedding_substitute import WordEmbeddingSubstitute
from .word_level.word_gradient_substitute import WordGradientSubstitute
from .word_level.word_masked_lm_substitute import WordMaskedLMSubstitute
from .word_level.word_change_loc_substitute import WordChangeLocSubstitute
from .word_level.word_change_name_substitute import WordChangeNameSubstitute
from .word_level.word_change_num_substitute import WordChangeNumSubstitute
from .word_level.word_extend_substitute import WordExtendSubstitute
from .word_level.word_contract_substitute import WordContractSubstitute
from .word_level.word_masked_lm_insertion import WordMaskedLMInsertion
from .word_level.word_masked_lm_merge import WordMaskedLMMerge
from .word_level.word_embedding_gradient_sign_substitute import (
    WordEmbeddingGradientSignSubstitute,
)

from .char_level.char_dces_substitute import CharacterDCESSubstitute
from .char_level.char_homoglyph_substitute import CharacterHomoglyphSubstitute
from .char_level.nbh_char_substitute import NeighboringCharacterSubstitute
from .char_level.random_char_deletion import RandomCharacterDeletion
from .char_level.random_char_insertion import RandomCharacterInsertion
from .char_level.random_char_substitute import RandomCharacterSubstitute
from .char_level.chinese_fyh_char import ChineseFYHCharSubstitute
from .char_level.char_qwerty_substitute import CharacterQWERTYSubstitute


__all__ = [
    "Transformation",
    "CharSubstitute",
    "WordSubstitute",
    "WordInsertion",
    "WordDeletion",
    "WordMerge",
    "PhraseInsertion",
    "CompositeTransformation",
    "RandomCompositeTransformation",
    "transformation_consists_of",
    "transformation_consists_of_word_substitutes",
    "transformation_consists_of_word_substitutes_and_deletions",
    "ChineseCiLinSubstitute",
    "ChineseWordNetSubstitute",
    "ChineseHowNetSubstitute",
    "WordNetSubstitute",
    "WordHowNetSubstitute",
    "WordEmbeddingSubstitute",
    "WordGradientSubstitute",
    "WordMaskedLMSubstitute",
    "WordChangeLocSubstitute",
    "WordChangeNameSubstitute",
    "WordChangeNumSubstitute",
    "WordExtendSubstitute",
    "WordContractSubstitute",
    "WordMaskedLMInsertion",
    "WordMaskedLMMerge",
    "WordEmbeddingGradientSignSubstitute",
    "CharacterDCESSubstitute",
    "CharacterHomoglyphSubstitute",
    "NeighboringCharacterSubstitute",
    "RandomCharacterDeletion",
    "RandomCharacterInsertion",
    "RandomCharacterSubstitute",
    "ChineseFYHCharSubstitute",
    "CharacterQWERTYSubstitute",
]
