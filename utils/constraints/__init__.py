# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-23
@LastEditTime: 2021-09-26
"""

from .base import Constraint, PreTransformationConstraint

from .edit_distance import EditDistance
from .part_of_speech import PartOfSpeech
from .grammar_error import GrammarError
from .word_embedding_distance import WordEmbeddingDistance
from .max_words_perturbed import MaxWordsPerturbed

from .sentence_encoder_constraints.sentence_encoder_base import SentenceEncoderBase
from .sentence_encoder_constraints.use_multilingual import MultilingualUSE
from .sentence_encoder_constraints.thought_vector import ThoughtVector

# from .sentence_encoder_constraints.use_xling_many import XlingManyUSE  # 暂时不使用
from .sentence_encoder_constraints.bert import BERT
from .sentence_encoder_constraints.infer_sent import InferSent

from .language_model_constraints.language_model_base import LanguageModelBase
from .language_model_constraints.google_language_model import GoogleLanguageModel
from .language_model_constraints.learning2write import Learning2Write
from .language_model_constraints.gpt2 import GPT2

from .input_column_modification import InputColumnModification
from .repeat_modification import RepeatModification
from .stopword_modification import StopwordModification
from .min_word_len import MinWordLen
from .max_modification_rate import MaxModificationRate


__all__ = [
    "Constraint",
    "PreTransformationConstraint",
    "EditDistance",
    "GrammarError",
    "PartOfSpeech",
    "WordEmbeddingDistance",
    "MaxWordsPerturbed",
    "SentenceEncoderBase",
    "MultilingualUSE",
    # "XlingManyUSE",
    "ThoughtVector",
    "BERT",
    "InferSent",
    "LanguageModelBase",
    "GoogleLanguageModel",
    "Learning2Write",
    "GPT2",
    "InputColumnModification",
    "RepeatModification",
    "StopwordModification",
    "MinWordLen",
    "MaxModificationRate",
]
