# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-31
@LastEditTime: 2022-03-19

日志基类
"""

import sys
import logging
from abc import ABC, abstractmethod
from typing import Iterable, NoReturn

from ...EvalBox.Attack.attack_result import AttackResult
from ...utils.strings import ReprMixin


__all__ = [
    "AttackLogger",
]


class AttackLogger(ReprMixin, ABC):
    """An abstract class for different methods of logging attack results."""

    __name__ = "AttackLogger"

    def __init__(self) -> NoReturn:
        self._default_logger = logging.getLogger("NLPAttack")
        self._default_logger.setLevel(logging.INFO)
        self._init_stream_handler()

    def _init_stream_handler(self) -> NoReturn:
        """ """
        if any(
            [
                isinstance(h, logging.StreamHandler)
                for h in self._default_logger.handlers
            ]
        ):
            return
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("%(name)s - %(levelname)s -\n %(message)s")
        c_handler.setFormatter(c_format)
        self._default_logger.addHandler(c_handler)

    @abstractmethod
    def log_attack_result(
        self, result: AttackResult, examples_completed: int
    ) -> NoReturn:
        raise NotImplementedError

    def log_summary_rows(self, rows: Iterable, title: str, window_id: str) -> NoReturn:
        pass

    def log_hist(
        self, arr: Iterable, numbins: int, title: str, window_id: str
    ) -> NoReturn:
        pass

    def log_sep(self) -> NoReturn:
        pass

    def flush(self) -> NoReturn:
        for h in self._default_logger.handlers:
            h.flush()
        self._flushed = True

    def close(self) -> NoReturn:
        self._clear_default_logger_handlers()
        del self._default_logger
        logging.shutdown()

    def _clear_default_logger_handlers(self) -> NoReturn:
        """ """
        for h in self._default_logger.handlers:
            h.close()
            self._default_logger.removeHandler(h)
