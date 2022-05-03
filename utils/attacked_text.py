# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-03
@LastEditTime: 2022-04-15

文本对抗攻击的原样本、对抗样本的类，包含切词，POS等基本操作，
主要基于TextAttack的AttackedText类进行的实现

"""

import math
import re
import warnings
from collections import OrderedDict, Counter
from copy import deepcopy
from typing import Union, Optional, Sequence, List, Set, NoReturn, Dict, Any

import numpy as np
import torch

from .strings import (  # noqa: F401
    words_from_text,
    color_text,
    LANGUAGE,
    normalize_language,
    UNIVERSAL_POSTAG,
    NERTAG,
    normalize_ner_tag,
)
from .strings_cn import words_from_text_cn, jieba_tag
from .strings_en import (  # noqa: F401
    flair_tag,
    nltk_tag,
    stanza_tag,
    tokenize,
    remove_space_before_punct,
)


__all__ = [
    "AttackedText",
    "ChineseAttackedText",
]


class AttackedText(object):
    """
    `AttackedText` instances that were perturbed from other `AttackedText`
    objects contain a pointer to the previous text
    (`attack_attrs["previous_attacked_text"]`), so that the full chain of
    perturbations might be reconstructed by using this key to form a linked list.
    """

    SPLIT_TOKEN = "<SPLIT>"
    __name__ = "AttackedText"

    def __init__(
        self,
        language: str,
        text_input: Union[str, dict],
        attack_attrs: dict = None,
        **kwargs: Any,
    ) -> NoReturn:
        """
        @param {
            language: 文本的语言，当前只支持中文或英文
            text_input: 当前实例的文本
            attack_attrs: 一次对抗攻击过程中需要保存的中间变量
        }
        @return: None
        """
        self.language = normalize_language(language)
        if self.language == LANGUAGE.CHINESE:
            import jieba

            self._tagger = jieba
        elif self.language == LANGUAGE.ENGLISH:
            self._tagger = nltk_tag

        if isinstance(text_input, str):
            self._text_input = OrderedDict(
                [("text", remove_space_before_punct(text_input))]
            )
        elif isinstance(text_input, OrderedDict):
            # self._text_input = deepcopy(text_input)
            self._text_input = OrderedDict(
                {k: remove_space_before_punct(v) for k, v in text_input.items()}
            )
        else:
            raise TypeError(
                f"text_input的类型`{type(text_input).__name__}`为无效的类型。有效的类型为 str 或 OrderedDict"
            )
        self._words = None
        self._words_per_input = None
        self._words_frozen = None
        self._pos_tags = None
        self._ner_tags = None

        self._max_eq_compare_depth = kwargs.get("max_eq_compare_depth", 3)

        if attack_attrs is None:
            self.attack_attrs = dict()
        elif isinstance(attack_attrs, dict):
            self.attack_attrs = attack_attrs
        else:
            raise TypeError(
                f"attack_attrs的类型 {type(attack_attrs).__name__}为无效的类型。有效的类型为 dict"
            )
        # Indices of words from the *original* text. Allows us to map
        # indices between original text and this text, and vice-versa.
        self.attack_attrs.setdefault("original_index_map", np.arange(self.num_words))
        # A list of all indices in *this* text that have been modified.
        self.attack_attrs.setdefault("modified_indices", set())
        self._word_sep = {
            LANGUAGE.ENGLISH: " ",
            LANGUAGE.CHINESE: "",
        }

    def __eq__(self, other: "AttackedText", depth: int = 0) -> bool:
        """当 self 与 other 文本相同，且有相同 attack_attrs 时，认为二者是相等的

        Since some elements stored in `self.attack_attrs` may be numpy
        arrays, we have to take special care when comparing them.

        NOTE:
        遇到了一下的报错
        ```python
        File "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/utils/attacked_text.py", line 116, in __eq__
            if not self.attack_attrs[key] == other.attack_attrs[key]:
        File "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/utils/attacked_text.py", line 116, in __eq__
            if not self.attack_attrs[key] == other.attack_attrs[key]:
        File "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/utils/attacked_text.py", line 116, in __eq__
            if not self.attack_attrs[key] == other.attack_attrs[key]:
        [Previous line repeated 242 more times]
        File "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/utils/attacked_text.py", line 103, in __eq__
            if not (self.text == other.text):
        RecursionError: maximum recursion depth exceeded in comparison
        ```
        解决方法：添加`depth`参数，默认为0，当depth超过预设值时，返回False. 例如
        ```python
        at1 = AttackedText("zh", "我是你爸爸的好哥们儿。")
        at2 = AttackedText("zh", "我是你爸爸的好哥们儿。")
        at1.attack_attrs["hehe"] = AttackedText("zh", "我是你爹的好哥们儿。")
        at2.attack_attrs["hehe"] = AttackedText("zh", "我是你爹的好哥们儿。")
        ```
        `at1.__eq__(at2,depth=2)`为`True`, `at1.__eq__(at2,depth=3)`为`False`
        """
        if depth > self._max_eq_compare_depth:
            return False
        if type(self) != type(other):
            return False
        if not (self.text == other.text):
            return False
        if len(self.attack_attrs) != len(other.attack_attrs):
            return False
        for key in self.attack_attrs:
            if key not in other.attack_attrs:
                return False
            elif isinstance(self.attack_attrs[key], np.ndarray):
                if not (self.attack_attrs[key].shape == other.attack_attrs[key].shape):
                    return False
                elif not (self.attack_attrs[key] == other.attack_attrs[key]).all():
                    return False
            elif isinstance(self.attack_attrs[key], AttackedText):
                if not self.attack_attrs[key].__eq__(
                    other.attack_attrs[key], depth + 1
                ):
                    return False
            else:
                if not self.attack_attrs[key] == other.attack_attrs[key]:
                    return False
        return True

    def __hash__(self) -> int:
        return hash(self.text)

    def free_memory(self) -> NoReturn:
        """Delete items that take up memory.

        Can be called once the AttackedText is only needed to display.
        """
        if "previous_attacked_text" in self.attack_attrs:
            self.attack_attrs["previous_attacked_text"].free_memory()
            self.attack_attrs.pop("previous_attacked_text", None)

        self.attack_attrs.pop("last_transformation", None)

        for key in self.attack_attrs:
            if isinstance(self.attack_attrs[key], torch.Tensor):
                self.attack_attrs.pop(key, None)

    def text_window_around_index(self, index: int, window_size: int) -> str:
        """The text window of `window_size` words centered around `index`."""
        length = self.num_words
        half_size = (window_size - 1) / 2.0
        if index - half_size < 0:
            start = 0
            end = min(window_size - 1, length - 1)
        elif index + half_size >= length:
            start = max(0, length - window_size)
            end = length - 1
        else:
            start = index - math.ceil(half_size)
            end = index + math.floor(half_size)
        text_idx_start = self._text_index_of_word_index(start)
        text_idx_end = self._text_index_of_word_index(end) + len(self.words[end])
        return self.text[text_idx_start:text_idx_end]

    def pos_of_word_index(self, word_idx: int) -> str:
        """计算第 `word_idx` 个词的词性（POS)"""
        if self.language == LANGUAGE.ENGLISH:
            return self._pos_of_word_index_en(word_idx)
        if not self._pos_tags:
            # self._pos_tags = jieba_tag(self.text, "pos")
            self._pos_tags = jieba_tag(" ".join(self.words), "pos")
            if set(self.words) == set(self._pos_tags["words"]):
                warnings.warn("jieba分词结果与当前实例words集合不一致")
                self._pos_tags = None
                return UNIVERSAL_POSTAG.OTHER
        pos = self._pos_tags["tags"][word_idx]
        return pos

    def ner_of_word_index(self, word_idx: int) -> str:
        """计算第 `word_idx` 个词的实体识别结果（NER）"""
        if self.language == LANGUAGE.ENGLISH:
            return self._ner_of_word_index_en(word_idx)
        if not self._ner_tags:
            self._ner_tags = jieba_tag(self.text, "ner")
            if set(self.words) == set(self._ner_tags["words"]):
                warnings.warn("jieba分词结果与当前实例words集合不一致")
                self._ner_tags = None
                return NERTAG("OTHER")
        ner = self._ner_tags["tags"][word_idx]
        return ner

    def _pos_of_word_index_en(self, word_idx: int) -> str:
        """计算英文文本的第 `word_idx` 个词的词性（POS)，
        这里需要检查得到的单词列表与`word_from_text`得到的单词列表是否相同
        """
        if not self._pos_tags:
            self._pos_tags = nltk_tag(self.text)
        word_list = self._pos_tags["words"]
        pos_list = self._pos_tags["tags"]

        for idx, word in enumerate(self.words):
            assert (
                word in word_list
            ), f"word `{word}`` absent in returned part-of-speech tags {self._pos_tags}"
            word_idx_in_tags = word_list.index(word)
            if idx == word_idx:
                return pos_list[word_idx_in_tags]
            else:
                word_list = word_list[word_idx_in_tags + 1 :]
                pos_list = pos_list[word_idx_in_tags + 1 :]

        raise ValueError(f"Did not find word from index {word_idx} in POS tag")

    def _ner_of_word_index_en(self, word_idx: int) -> str:
        """计算英文文本的第 `word_idx` 个词的实体识别结果（NER）"""
        if not self._ner_tags:
            self._ner_tags = flair_tag(self.text, "ner-large")
        word_list = self._ner_tags["words"]
        ner_list = self._ner_tags["tags"]

        for idx, word in enumerate(word_list):
            word_idx_in_tags = word_list.index(word)
            if idx == word_idx:
                return ner_list[word_idx_in_tags]
            else:
                word_list = word_list[word_idx_in_tags + 1 :]
                ner_list = ner_list[word_idx_in_tags + 1 :]

        raise ValueError(f"Did not find word from index {word_idx} in NER tag")

    def _text_index_of_word_index(self, word_idx: int) -> int:
        """计算第 `word_idx` 个词在目前文本中的下标(以字符计)"""
        pre_words = self.words[: word_idx + 1]
        lower_text = self.text.lower()
        # Find all words until `word_idx` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word.lower(), look_after_index) + len(
                word
            )
        look_after_index -= len(self.words[word_idx])
        return look_after_index

    def text_until_word_index(self, word_idx: int) -> str:
        """计算从整个文本开头到第 `word_idx` 个词之前的文本字符串"""
        look_after_index = self._text_index_of_word_index(word_idx)
        return self.text[:look_after_index]

    def text_after_word_index(self, word_idx: int) -> str:
        """计算从第 `word_idx` 个词之后到整个文本末尾的文本字符串"""
        # Get index of beginning of word then jump to end of word.
        look_after_index = self._text_index_of_word_index(word_idx) + len(
            self.words[word_idx]
        )
        return self.text[look_after_index:]

    def first_word_diff(self, other: "AttackedText") -> str:
        """计算当前词序列与 `other` `AttackedText` 的词序列不同的第一个元素（词）

        Useful for word swap strategies.
        """
        for i in range(min(len(self.words), len(other.words))):
            if self.words[i] != other.words[i]:
                return self.words[i]
        return None

    def first_word_diff_index(self, other: "AttackedText") -> int:
        """计算当前词序列与 `other` `AttackedText` 的词序列不同的第一个元素（词）的下标

        Useful for word swap strategies.
        """
        w1 = self.words
        w2 = other.words
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return i
        return None

    def all_words_diff(self, other: "AttackedText") -> Set[int]:
        """计算当前词序列与`other` `AttackedText`词序列元素值（词）不一致的下标的集合"""
        indices = set()
        for i in range(min(len(self.words), len(other.words))):
            if self.words[i] != other.words[i]:
                indices.add(i)
        return indices

    def words_diff_num(self, other: "AttackedText") -> int:
        # using edit distance to calculate words diff num
        # TODO: directly use `Levenshtein.distance` to compute
        def generate_tokens(words):
            result = dict()
            idx = 1
            for w in words:
                if w not in result:
                    result[w] = idx
                    idx += 1
            return result

        def words_to_tokens(words, tokens):
            result = []
            for w in words:
                result.append(tokens[w])
            return result

        def edit_distance(w1_t, w2_t):
            matrix = [
                [i + j for j in range(len(w2_t) + 1)] for i in range(len(w1_t) + 1)
            ]

            for i in range(1, len(w1_t) + 1):
                for j in range(1, len(w2_t) + 1):
                    if w1_t[i - 1] == w2_t[j - 1]:
                        d = 0
                    else:
                        d = 1
                    matrix[i][j] = min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + d,
                    )

            return matrix[len(w1_t)][len(w2_t)]

        def cal_dif(w1, w2):
            tokens = generate_tokens(w1 + w2)
            w1_t = words_to_tokens(w1, tokens)
            w2_t = words_to_tokens(w2, tokens)
            return edit_distance(w1_t, w2_t)

        w1 = self.words
        w2 = other.words
        return cal_dif(w1, w2)

    def ith_word_diff(self, other: "AttackedText", word_idx: int) -> bool:
        """判断当前词序列第 `word_idx` 个词是否与 `other` `AttackedText` 第 `word_idx` 个词是不同的"""
        w1 = self.words
        w2 = other.words
        if len(w1) - 1 < word_idx or len(w2) - 1 < word_idx:
            return True
        return w1[word_idx] != w2[word_idx]

    def convert_from_original_idxs(self, indices: Sequence[int]) -> List[int]:
        """将下标在序列`indices`中的原文本中的词，转换为它们在当前文本中词序列中的下标

        方法：利用`self.attack_attrs['original_index_map']`，进行转换
        """
        if len(self.attack_attrs["original_index_map"]) == 0:
            return indices
        elif isinstance(indices, set):
            indices = list(indices)

        elif not isinstance(indices, [list, np.ndarray]):
            raise TypeError(f"输入的 indices 的类型 {type(indices).__name__} 为不合法的类型")

        return [self.attack_attrs["original_index_map"][i] for i in indices]

    def replace_words_at_indices(
        self, indices: Sequence[int], new_words: Sequence[str]
    ) -> "AttackedText":
        """创建新`AttackedText`实例，并
        将当前文本下标在序列`indices`中的词替换为序列`new_words`中的元素（词），
        赋值到新创建的`AttackedText`实例中，并返回该实例
        """
        if len(indices) != len(new_words):
            raise ValueError(
                f"`new_words`序列长度（={len(new_words)}）与`indices`的长度（={len(indices)}）不一致"
            )
        assert all(
            [isinstance(new_word, str) for new_word in new_words]
        ), "`new_words` 序列中元素要求都是 str 类型"
        words = self.words[:]
        for i, new_word in zip(indices, new_words):
            if (i < 0) or (i > len(words)):
                raise ValueError(f"`indices`中的下标{i}越界了")
            words[i] = new_word
        return self.generate_new_attacked_text(words)

    def replace_word_at_index(self, index: int, new_word: str) -> "AttackedText":
        """创建新`AttackedText`实例，并
        将当前文本下标为`index`中的词替换为`new_word`，
        赋值到新创建的`AttackedText`实例中，并返回该实例"""
        if not isinstance(new_word, str):
            raise TypeError(f"`new_word`合法类型为 str, 输入类型为 {type(new_word).__name__}")
        return self.replace_words_at_indices([index], [new_word])

    def delete_word_at_index(self, index: int) -> "AttackedText":
        """创建新`AttackedText`实例，并
        将当前文本下标为`index`中的词删除，并返回该实例"""
        return self.replace_word_at_index(index, "")

    def insert_text_after_word_index(self, index: int, text: str) -> "AttackedText":
        """创建新`AttackedText`实例，并
        将当前文本下标为`index`中的词后添加文本 `text`，并返回该实例"""
        if not isinstance(text, str):
            raise TypeError(f"text`合法类型为 str, 输入类型为 {type(text).__name__}")
        if index == -len(self.words) - 1 and len(self.words) > 0:
            return self.insert_text_before_word_index(0, text)
        word_at_index = self.words[index]
        new_text = self._word_sep[self.language].join((word_at_index, text))
        return self.replace_word_at_index(index, new_text)

    def insert_text_before_word_index(self, index: int, text: str) -> "AttackedText":
        """创建新`AttackedText`实例，并
        将当前文本下标为`index`中的词前添加文本 `text`，并返回该实例"""
        if not isinstance(text, str):
            raise TypeError(f"text must be an str, got type {type(text)}")
        if index == len(self.words):
            return self.insert_text_after_word_index(-1, text)
        word_at_index = self.words[index]
        # TODO if `word_at_index` is at the beginning of a sentence, we should
        # optionally capitalize `text`.
        new_text = self._word_sep[self.language].join((text, word_at_index))
        return self.replace_word_at_index(index, new_text)

    def get_deletion_indices(self) -> List[int]:
        """计算当前文本相较于原文本被删除词的下标"""
        return self.attack_attrs["original_index_map"][
            self.attack_attrs["original_index_map"] == -1
        ]

    def generate_new_attacked_text(self, new_words: Sequence[str]) -> "AttackedText":
        """
        创建新`AttackedText`实例，将当前实例的词序列`self.words` 替换为 `new_words`,
        赋值于新实例。

        注意，`new_words` 中的元素可能是空字符串（删除词）；
        可能是多个词组成的短语（插入词、短语）
        """
        perturbed_text = ""
        original_text = AttackedText.SPLIT_TOKEN.join(self._text_input.values())

        new_attack_attrs = dict()
        if "label_names" in self.attack_attrs:
            new_attack_attrs["label_names"] = self.attack_attrs["label_names"]
        new_attack_attrs["newly_modified_indices"] = set()
        # Point to previously monitored text.
        new_attack_attrs["previous_attacked_text"] = self
        # Use `new_attack_attrs` to track indices with respect to the original
        # text.
        new_attack_attrs["modified_indices"] = self.attack_attrs[
            "modified_indices"
        ].copy()  # type: set
        new_attack_attrs["original_index_map"] = self.attack_attrs[
            "original_index_map"
        ].copy()  # type: np.ndarray

        new_i = 0
        # Create the new attacked text by swapping out words from the original
        # text with a sequence of 0+ words in the new text.
        for i, (input_word, adv_word_seq) in enumerate(zip(self.words, new_words)):
            try:
                word_start = original_text.index(input_word)
            except Exception as e:
                print(
                    f"input_word `{input_word}` is not a substring of original_text `{original_text}`"
                )
                raise e
            word_end = word_start + len(input_word)
            perturbed_text += original_text[:word_start]
            original_text = original_text[word_end:]
            adv_words = self._words_from_text(adv_word_seq)
            adv_num_words = len(adv_words)
            num_words_diff = adv_num_words - len(self._words_from_text(input_word))
            # Track indices on insertions and deletions.
            if num_words_diff != 0:
                # Re-calculated modified indices. If words are inserted or deleted,
                # they could change.
                shifted_modified_indices = set()
                for modified_idx in new_attack_attrs["modified_indices"]:
                    if modified_idx < i:
                        shifted_modified_indices.add(modified_idx)
                    elif modified_idx > i:
                        shifted_modified_indices.add(modified_idx + num_words_diff)
                    else:
                        pass
                new_attack_attrs["modified_indices"] = shifted_modified_indices
                # Track insertions and deletions wrt original text.
                # original_modification_idx = i
                new_idx_map = new_attack_attrs["original_index_map"].copy()
                if num_words_diff == -1:
                    # Word deletion
                    new_idx_map[new_idx_map == i] = -1
                new_idx_map[new_idx_map > i] += num_words_diff

                if num_words_diff > 0 and input_word != adv_words[0]:
                    # If insertion happens before the `input_word`
                    new_idx_map[new_idx_map == i] += num_words_diff

                new_attack_attrs["original_index_map"] = new_idx_map
            # Move pointer and save indices of new modified words.
            for j in range(i, i + adv_num_words):
                if input_word != adv_word_seq:
                    new_attack_attrs["modified_indices"].add(new_i)
                    new_attack_attrs["newly_modified_indices"].add(new_i)
                new_i += 1
            # Check spaces for deleted text.
            if adv_num_words == 0 and len(original_text):
                # Remove extra space (or else there would be two spaces for each
                # deleted word).
                # @TODO What to do with punctuation in this case? This behavior is undefined.
                if i == 0:
                    # If the first word was deleted, take a subsequent space.
                    if original_text[0] == " ":
                        original_text = original_text[1:]
                else:
                    # If a word other than the first was deleted, take a preceding space.
                    if perturbed_text[-1] == " ":
                        perturbed_text = perturbed_text[:-1]
            # Add substitute word(s) to new sentence.
            perturbed_text += adv_word_seq
        perturbed_text += original_text  # Add all of the ending punctuation.
        # Reform perturbed_text into an OrderedDict.
        perturbed_text = re.sub("[\\s]+", " ", perturbed_text).strip()
        perturbed_input_texts = perturbed_text.split(AttackedText.SPLIT_TOKEN)
        perturbed_input = OrderedDict(
            zip(self._text_input.keys(), perturbed_input_texts)
        )
        return AttackedText(
            self.language, perturbed_input, attack_attrs=new_attack_attrs
        )

    def words_diff_ratio(self, x: "AttackedText") -> "AttackedText":
        """计算当前文本与 `x`之间的词替换比例。

        Note that current text and `x` must have same number of words.
        """
        # assert self.num_words == x.num_words, "当前文本应与输入文本`x`有相同数目的词"
        if self.num_words != x.num_words:
            warnings.warn("当前文本应与输入文本`x`有不同数目的词")
        compare_len = min(self.num_words, x.num_words)
        return (
            float(
                np.sum(
                    np.array(self.words)[:compare_len]
                    != np.array(x.words)[:compare_len]
                )
            )
            / self.num_words
        )

    def align_with_model_tokens(self, model_wrapper) -> Dict[int, List[int]]:
        """ """
        tokens = model_wrapper.tokenize([self.tokenizer_input], strip_prefix=True)[0]
        word2token_mapping = {}
        j = 0
        last_matched = 0

        for i, word in enumerate(self.words):
            matched_tokens = []
            while j < len(tokens) and len(word) > 0:
                token = tokens[j].lower()
                idx = word.lower().find(token)
                if idx == 0:
                    word = word[idx + len(token) :]
                    matched_tokens.append(j)
                    last_matched = j
                j += 1

            if not matched_tokens:
                word2token_mapping[i] = None
                j = last_matched
            else:
                word2token_mapping[i] = matched_tokens

        return word2token_mapping

    def _words_from_text(self, text: Optional[str] = None) -> List[str]:
        """ """
        _text = text if text is not None else self.text
        if self.language == LANGUAGE.ENGLISH:
            # words = words_from_text(_text)
            words = tokenize(_text, backend="naive")
        elif self.language == LANGUAGE.CHINESE:
            words = words_from_text_cn(_text)
        return words

    @property
    def tokenizer_input(self) -> tuple:
        """The tuple of inputs to be passed to the tokenizer."""
        input_tuple = tuple(self._text_input.values())
        # Prefer to return a string instead of a tuple with a single value.
        if len(input_tuple) == 1:
            return input_tuple[0]
        else:
            return input_tuple

    @property
    def column_labels(self) -> List[str]:
        """Returns the labels for this text's columns.

        For single-sequence inputs, this simply returns ['text'].
        """
        return list(self._text_input.keys())

    @property
    def words_per_input(self) -> List[List[str]]:
        """Returns a list of lists of words corresponding to each input."""
        if not self._words_per_input:
            self._words_per_input = [
                self._words_from_text(_input) for _input in self._text_input.values()
            ]
        return self._words_per_input

    @property
    def words(self) -> List[str]:
        if not self._words:
            self._words = self._words_from_text()
            self._words_frozen = deepcopy(self._words)
        return self._words

    @property
    def text(self) -> str:
        """Represents full text input.

        Multiply inputs are joined with a line break.
        """
        return "\n".join(self._text_input.values())

    @property
    def num_words(self) -> int:
        """Returns the number of words in the sequence."""
        return len(self.words)

    @property
    def word_count(self) -> Dict[str, int]:
        """Returns a dictionary of word counts.

        The keys are the words, and the values are the counts.
        """
        return dict(Counter(self.words))

    def printable_text(
        self, key_color: str = "bold", key_color_method: str = None
    ) -> str:
        """Represents full text input. Adds field descriptions.

        For example, entailment inputs look like:
            ``
            premise: ...
            hypothesis: ...
            ``
        """
        # For single-sequence inputs, don't show a prefix.
        if len(self._text_input) == 1:
            return next(iter(self._text_input.values()))
        # For multiple-sequence inputs, show a prefix and a colon. Optionally,
        # color the key.
        else:
            if key_color_method:

                def ck(k):
                    return color_text(k, key_color, key_color_method)

            else:

                def ck(k):
                    return k

            return "\n".join(
                f"{ck(key.capitalize())}: {value}"
                for key, value in self._text_input.items()
            )

    def __repr__(self) -> str:
        return f'<{self.__name__} "{self.text}">'

    def __str__(self) -> str:
        return self.__repr__()


class ChineseAttackedText(AttackedText):
    """ """

    __name__ = "ChineseAttackedText"

    def __init__(self, text_input: Union[str, dict], attack_attrs: dict = None):
        """
        @param {
            text_input: 当前实例的文本
            attack_attrs: 一次对抗攻击过程中需要保存的中间变量
        }
        @return: None
        """
        super().__init__("cn", text_input, attack_attrs)
