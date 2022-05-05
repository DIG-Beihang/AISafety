# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-03
@LastEditTime: 2021-08-18
"""

import language_tool_python

from .base import Constraint
from ..attacked_text import AttackedText
from ..strings import LANGUAGE, normalize_language


__all__ = [
    "GrammarError",
]


class GrammarError(Constraint):
    """
    Grammar error number constraint

    使用 `languagetool` (`language_tool_python`)检查语法错误情况，
    即原样本与对抗样本是否有同样数量的语法错误。
    """

    def __init__(
        self,
        language: str = "zh",
        grammar_error_threshold: int = 0,
        compare_against_original: bool = True,
    ):
        """
        @description: Grammar error number constraint
        @param {
            language:
            grammar_error_threshold:
            compare_against_original:
        }
        @return: None
        """
        super().__init__(compare_against_original)
        self.grammar_error_threshold = grammar_error_threshold
        self.grammar_error_cache = {}
        self.set_language(language)

    def set_language(self, language: str):
        """ """
        self._language = normalize_language(language)
        if self._language == LANGUAGE.CHINESE:
            self.lang_tool = language_tool_python.LanguageTool("zh-CN")
        elif self._language == LANGUAGE.ENGLISH:
            self.lang_tool = language_tool_python.LanguageTool("en-US")

    def get_num_errors(
        self, attacked_text: AttackedText, use_cache: bool = False
    ) -> int:
        """ """
        text = attacked_text.text
        if use_cache:
            if text not in self.grammar_error_cache:
                self.grammar_error_cache[text] = len(self.lang_tool.check(text))
            return self.grammar_error_cache[text]
        else:
            return len(self.lang_tool.check(text))

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """ """
        original_num_errors = self.get_num_errors(reference_text, use_cache=True)
        errors_added = self.get_num_errors(transformed_text) - original_num_errors
        return errors_added <= self.grammar_error_threshold
