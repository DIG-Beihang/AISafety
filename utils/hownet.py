# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2022-03-19

主要修改自
1. https://github.com/thunlp/OpenHowNet/blob/master/OpenHowNet/Standards.py
2. https://github.com/thunlp/OpenHowNet/blob/master/OpenHowNet/SememeTreeParser.py

OpenHowNet全量太大，考虑不使用
"""

import os
import pickle
from typing import NoReturn, List, Optional, Union

from anytree import Node
from anytree.exporter import DictExporter

from .misc import nlp_cache_dir
from .strings import normalize_language, LANGUAGE, ReprMixin


__all__ = [
    "HowNetDict",
]


class HowNetDict(ReprMixin):
    """ """

    __NAME_CHOICES = [
        "name_en",
        "name_ch",
        "all",
    ]
    __name__ = "HowNetDict"

    def __init__(self) -> NoReturn:
        """ """
        try:
            data_dir = os.path.join(
                nlp_cache_dir, "openhownet_data", "HowNet_dict_complete"
            )
            # load dict complete
            with open(data_dir, "rb") as origin_dict:
                word_dict = pickle.load(origin_dict)
        except FileNotFoundError as e:
            # TODO: downloading from
            # https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenHowNet/openhownet_data.zip
            print("暂不支持在线加载")
            print(e)

        self.en_map = dict()
        self.zh_map = dict()
        self.ids = dict()

        for key in word_dict:
            now_dict = word_dict[key]
            en_word = now_dict["en_word"].strip()
            zh_word = now_dict["ch_word"].strip()
            if en_word not in self.en_map:
                self.en_map[en_word] = list()
            self.en_map[en_word].append(now_dict)
            if zh_word not in self.zh_map:
                self.zh_map[zh_word] = list()
            self.zh_map[zh_word].append(now_dict)
            if now_dict["No"] not in self.ids:
                self.ids[now_dict["No"]] = list()
            self.ids[now_dict["No"]].append(now_dict)

    def __getitem__(self, item: str) -> List[dict]:
        """ """
        res = list()
        if item in self.en_map:
            res.extend(self.en_map[item])
        if item in self.zh_map:
            res.extend(self.zh_map[item])
        if item in self.ids:
            res.extend(self.ids[item])
        return res

    def __len__(self) -> int:
        return len(self.ids)

    def get(
        self, word: str, language: Optional[Union[str, LANGUAGE]] = None
    ) -> List[dict]:
        """ """
        res = list()
        if not language:
            return self[word]
        _language = normalize_language(language)
        if _language == LANGUAGE.ENGLISH:
            if word in self.en_map:
                res = self.en_map[word]
        elif _language == LANGUAGE.CHINESE:
            if word in self.zh_map:
                res = self.zh_map[word]
        else:
            raise ValueError(f"不支持语言 {language}")
        return res

    def get_zh_words(self) -> List[str]:
        """ """
        return list(self.zh_map.keys())

    def get_en_words(self) -> List[str]:
        """ """
        return list(self.en_map.keys())

    def _expand_tree(
        self,
        tree: Union[dict, Node],
        propertyName: str,
        layer: int,
        isRoot: bool = True,
    ) -> set:
        """ """
        res = set()
        if layer == 0:
            return res
        target = tree

        # special process with the root node
        if isinstance(tree, dict):
            target = list()
            target.append(tree)
        for item in target:
            try:
                if not isRoot:
                    if propertyName not in HowNetDict.__NAME_CHOICES:
                        res.add(item[propertyName])
                    else:
                        choice = HowNetDict.__NAME_CHOICES.index(propertyName)
                        if choice < 2:
                            res.add(item["name"].split("|")[choice])
                        else:
                            res.add(item["name"])

                if "children" in item:
                    res |= self._expand_tree(
                        item["children"], propertyName, layer - 1, isRoot=False
                    )
            except Exception as e:
                # print("Bad Nodes:",item)
                if isinstance(e, IndexError):
                    continue
                raise e
        return res

    def get_sememes_by_word(
        self,
        word: str,
        structured: bool = False,
        language: Union[str, LANGUAGE] = "zh",
        merge: bool = False,
        expanded_layer: int = -1,
    ) -> Union[List[dict], dict]:
        """ """
        _language = normalize_language(language)
        queryResult = self[word]
        result = list()
        if structured:
            for item in queryResult:
                try:
                    result.append(
                        {"word": item, "tree": GenSememeTree(item["Def"], word)}
                    )
                except Exception as e:
                    print("Generate Sememe Tree Failed for", item["No"])
                    print("Exception:", e)
                    continue
        else:
            lang = {
                LANGUAGE.CHINESE: "ch",
                LANGUAGE.ENGLISH: "en",
            }[_language]

            name = lang + "_word"
            lang = "name_" + lang
            if merge:
                result = dict()
            for item in queryResult:
                try:
                    if not merge:
                        result.append(
                            {
                                "word": item[name],
                                "sememes": self._expand_tree(
                                    GenSememeTree(item["Def"], word),
                                    lang,
                                    expanded_layer,
                                ),
                            }
                        )
                    else:
                        if item[name] not in result:
                            result[item[name]] = set()
                        result[item[name]] |= set(
                            self._expand_tree(
                                GenSememeTree(item["Def"], word), lang, expanded_layer
                            )
                        )
                except Exception as e:
                    print(word)
                    print("Wrong Item:", item)
                    # print("Generate Sememe Tree Failed for", item["No"])
                    print("Exception:", e)
                    raise e
            if merge:
                if len(result.keys()) == 1:
                    key = list(result.keys())[0]
                    result = result[key]
        return result

    def has(self, item: str, language: Optional[Union[str, LANGUAGE]] = None) -> bool:
        """ """
        if not language:
            return item in self.en_map or item in self.zh_map or item in self.ids

        _language = normalize_language(language)
        if _language == LANGUAGE.ENGLISH:
            return item in self.en_map
        elif _language == LANGUAGE.CHINESE:
            return item in self.zh_map
        else:
            raise ValueError(f"不支持语言 {language}")


def trim_pattern(kdml: str) -> str:
    # remove RMK
    rmk_pos = kdml.find("RMK=")
    if rmk_pos >= 0:
        kdml = kdml[:rmk_pos]
    return kdml


def GenSememeTree(kdml: str, word: str, returnNode: bool = False) -> Union[Node, dict]:
    """输入义原描述字符串，返回义原结构树：dict形式"""
    # 将有";"符号的kdml拆分成多个子树
    kdml = trim_pattern(kdml)
    kdml_list = kdml.split(";")
    root = Node(word, role="sense")
    for kdml in kdml_list:

        entity_idx = []  # 义原起止位置集合
        node = []  # 树的节点集合
        pointer = []  # idx of "~" cases

        # 识别义原
        for i in range(len(kdml)):
            if kdml[i] in ["~", "?", "$"]:
                if kdml[i] == "~":
                    pointer.append(len(node))
                entity_idx.append([i, i + 1])
                node.append(Node(kdml[i], role="None"))
            elif kdml[i] == "|":
                start_idx = i
                end_idx = i
                while kdml[start_idx] not in ["{", '"']:
                    start_idx = start_idx - 1
                while kdml[end_idx] not in ["}", ":", '"']:
                    end_idx = end_idx + 1
                entity_idx.append([start_idx + 1, end_idx])
                node.append(Node(kdml[start_idx + 1 : end_idx], role="None"))
                # Dictionary.sememes.add(kdml[start_idx + 1: end_idx])
        for i in range(len(entity_idx)):
            cursor = entity_idx[i][0]
            left_brace = 0
            right_brace = 0
            quotation = 0
            # 找到当前义原所属的主义原位置
            while not (
                kdml[cursor] == ":"
                and (
                    (quotation % 2 == 0 and left_brace == right_brace + 1)
                    or (quotation % 2 == 1 and left_brace == right_brace)
                )
            ):
                if cursor == 0:
                    break
                if kdml[cursor] == "{":
                    left_brace = left_brace + 1
                elif kdml[cursor] == "}":
                    right_brace = right_brace + 1
                elif kdml[cursor] == '"':
                    quotation = quotation + 1
                cursor = cursor - 1
            parent_idx = -1
            for j in range(i - 1, -1, -1):  # 从当前位置往前找可以对应上的义原
                if entity_idx[j][1] == cursor:
                    node[i].parent = node[j]
                    parent_idx = j
                    break
            if i != 0:
                if parent_idx != -1:
                    right_range = entity_idx[parent_idx][1] - 1
                else:
                    right_range = entity_idx[i - 1][1] - 1
                role_begin_idx = -1
                role_end_idx = -1
                # 修改：在当前义原和父义原之间找
                for j in range(entity_idx[i][0] - 1, right_range, -1):
                    if kdml[j] == "=":
                        role_end_idx = j
                    elif kdml[j] in [",", ":"]:
                        role_begin_idx = j
                        break
                if role_end_idx != -1:
                    node[i].role = kdml[role_begin_idx + 1 : role_end_idx]
                    # Dictionary.roles.add(node[i].role)
        for i in pointer:
            node[i].parent.role = node[i].role
            node[i].parent = None
        node[0].parent = root

    # exporter = JsonExporter(indent=2, sort_keys=True)
    if not returnNode:
        # 转化成dict形式
        # exporter = DictExporter()
        return DictExporter().export(root)
    else:
        return root
