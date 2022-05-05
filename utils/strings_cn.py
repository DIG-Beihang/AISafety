# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-12
@LastEditTime: 2022-04-15

中文字符串的一些特有的一些操作，分词，POS标记等
"""

import re
from string import punctuation

import stanza
import jieba
import jieba.posseg as pseg
from zhon.hanzi import punctuation as zh_punctuation

from .strings import normalize_pos_tag, normalize_ner_tag

# Issue: 目前jieba在paddle模式下，分词有问题，会把标点符号分到下一个词上
# try:
#     jieba.enable_paddle()
#     _jieba_use_paddle = True
# except Exception:
#     _jieba_use_paddle = False
_jieba_use_paddle = False


__all__ = [
    "words_from_text_cn",
    "jieba_tag",
    "stanza_tag",
]


def words_from_text_cn(s: str, words_to_ignore: list = [], tol: int = 1) -> list:
    """

    目前，允许以空格分词，但是不能差 `jieba.lcut` 分词结果太远，
    否则还是采用 `jieba.lcut` 的分词。

    例子（以下是某个对抗样本的分词示例，以及原样本分词结果）：
    -------
    >>> words_from_text_cn("以后 还会 继续 在 杨晓卫 上 购物")
    ['以后', '还会', '继续', '在', '杨晓卫', '上', '购物']  # 7个词
    >>> words_from_text_cn("以后还会继续在杨晓卫上购物")
    ['以后', '还会', '继续', '在', '杨晓卫上', '购物']  # 6个词
    >>> words_from_text_cn("以后还会继续在京东上购物")  # 原样本
    ['以后', '还会', '继续', '在', '京东', '上', '购物']  # 7个词

    """
    pseudo_words = s.split(" ")
    words = jieba.lcut(
        re.sub("[\ufeff\\s]", "", s.strip()), use_paddle=_jieba_use_paddle
    )
    # words = list(filter(lambda w: re.search("\\w+", w), words))  # ignore empty words
    if abs(len(words) - len(pseudo_words)) <= tol:
        words = pseudo_words
    purifier = (
        lambda w: len(
            re.sub(
                f"([{punctuation+zh_punctuation}\\s]+)|{'|'.join(words_to_ignore)}",
                "",
                w.strip(),
            )
        )
        > 0
    )
    words = list(filter(purifier, words))
    return words


# fmt: off
_jieba_ner_mapping = {
    "nr": "PER", "PER": "PER",  # 人名
    "ns": "LOC", "LOC": "LOC",  # 地点
    "nt": "ORG", "ORG": "ORG",  # 机构
    "t": "TIME", "TIME": "TIME",  # 时间
    "nw": "WORK",  # 作品名
}
# fmt: on


def jieba_tag(s: str, tag_type: str = "pos") -> tuple:
    """ """
    assert tag_type.lower() in [
        "pos",
        "ner",
    ], f"""tag_type must be "pos" or "ner", but got {tag_type}"""
    purifier = (
        lambda w: len(re.sub(f"[{punctuation+zh_punctuation}\\s]+", "", w)) > 0
    )  # noqa: E731
    words, tags = [], []
    for item in pseg.cut(s.strip().replace("\ufeff", ""), use_paddle=_jieba_use_paddle):
        if not purifier(item.word):
            continue
        words.append(item.word)
        if tag_type.lower() == "ner":
            tags.append(
                normalize_ner_tag(
                    _jieba_ner_mapping.get(item.flag, "OTHER"),
                )
            )
        else:  # pos
            tags.append(normalize_pos_tag(item.flag))
    res = {"words": words, "tags": tags}
    return res


def stanza_tag(s: str, tag_type: str = "pos") -> dict:
    """ """
    raise NotImplementedError("stanza 需要下载大模型，暂时不用")
    stanza.Pipeline(
        lang="zh",
        processors="tokenize, pos",
        tokenize_pretokenized=True,
    )
