# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-12
@LastEditTime: 2022-03-19

英文字符串的一些基本操作
"""

import re
from string import punctuation
from typing import List, NoReturn

import nltk
from nltk.tokenize import word_tokenize as nltk_word_tokenize
import stanza
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.tokenizer import word_tokenizer as segtok_word_tokenize
from syntok.tokenizer import Tokenizer as syntokTokenizer

from .strings import (
    words_from_text,
    normalize_pos_tag,
    normalize_ner_tag,
)


__all__ = [
    "tokenize",
    "spacy_tag",
    "flair_tag",
    "nltk_tag",
    "stanza_tag",
    "remove_space_before_punct",
]


def _nltk_setup() -> NoReturn:
    """ """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
    try:
        nltk.data.find("taggers/universal_tagset")
    except LookupError:
        nltk.download("universal_tagset")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


_nltk_setup()


def tokenize(
    s: str, backend: str = "naive", ignore_punctuations: bool = True
) -> List[str]:
    """ """
    # _s = remove_space_before_punct(s)
    # TODO: deal with exceptions like `he 's`
    _s = s
    if backend.lower() == "nltk":
        words = nltk_word_tokenize(_s)
    elif backend.lower() == "segtok":
        words = segtok_word_tokenize(_s)
    elif backend.lower() == "syntok":
        tok = syntokTokenizer()
        words = [t.value for t in tok.tokenize(s)]
    elif backend.lower() == "naive":
        words = words_from_text(_s)
    if ignore_punctuations:
        words = list(
            filter(lambda w: len(re.sub(f"[{punctuation}]+", "", w)) > 0, words)
        )
        # words = list(filter(lambda w: w not in punctuation, words))
    return words


def flair_tag(s: str, tag_type: str = "flair/upos-english-fast") -> dict:
    """
    Tags a `flair` `Sentence` object using `flair` tagger.

    tag_type:
        "upos-english-fast": fast universal part-of-speech tagging model for English
        "ner-large": large named entity recognition model for English
        etc.

        ref. https://huggingface.co/flair
    """
    if not tag_type.startswith("flair/"):
        tag_type = "flair/" + tag_type.strip("/")
    sentence = Sentence(s, use_tokenizer=tokenize)
    tagger = SequenceTagger.load(tag_type)
    tagger.predict(sentence)
    words, tags = _zip_flair_result(sentence, tag_type)
    res = {"words": words, "tags": tags}
    return res


def _zip_flair_result(
    pred: Sentence, tag_type: str = "flair/upos-english-fast"
) -> tuple:
    """Takes a sentence tagging from `flair` and returns two lists, of words
    and their corresponding parts-of-speech."""

    if not isinstance(pred, Sentence):
        raise TypeError("Result from Flair POS tagger must be a `Sentence` object.")

    tokens = pred.tokens
    words = []
    tags = []
    for token in tokens:
        words.append(token.text)
        if "pos" in tag_type:
            pos_tag = token.get_tag("pos").value
            tags.append(normalize_pos_tag(pos_tag))
        elif tag_type == "ner":
            ner_tag = token.get_tag("ner")
            tags.append(normalize_ner_tag(ner_tag.value, ner_tag.score))
    return words, tags


def nltk_tag(s: str, tag_type: str = "pos") -> dict:
    """ """
    words = tokenize(s)
    if tag_type.lower() != "pos":
        raise ValueError("nltk 暂时只用于 POS tagging")
    tags = []
    for pair in nltk.pos_tag(words, tagset="universal", lang="eng"):
        tags.append(normalize_pos_tag(pair[1]))
    res = {"words": words, "tags": tags}
    return res


def spacy_tag(s: str, tag_type: str = "pos") -> dict:
    """ """
    raise NotImplementedError


def stanza_tag(s: str, tag_type: str = "pos") -> dict:
    """ """
    raise NotImplementedError("stanza 需要下载大模型，暂时不用")
    stanza.Pipeline(
        lang="en",
        processors="tokenize, pos",
        tokenize_pretokenized=True,
    )


def remove_space_before_punct(text: str) -> str:
    """ """
    pattern = f"[\\s]+[{punctuation}][\\w]+"
    start = 0
    res = ""
    for item in re.finditer(pattern, text):
        res += text[start : item.start()] + re.sub("\\s", "", item.group())
        start = item.end()
    res += text[start:]
    return res
