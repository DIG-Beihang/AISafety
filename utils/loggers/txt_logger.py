# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-31
@LastEditTime: 2021-11-11

文本文件日志
"""

import time
import os
import logging
from typing import NoReturn, Iterable, List, Optional, Any

import terminaltables

from .base import AttackLogger
from ..misc import nlp_log_dir
from ...EvalBox.Attack.attack_result import AttackResult


class TxtLogger(AttackLogger):
    """Logs the results of an attack to a file, or `stdout`."""

    __name__ = "TxtLogger"

    def __init__(
        self,
        filename: Optional[str] = None,
        stdout: bool = False,
        color_method: str = "ansi",
    ) -> NoReturn:
        """ """
        super().__init__()
        self.stdout = stdout
        self.filename = filename or os.path.join(
            nlp_log_dir, f"""{time.strftime("TxtLogger-%Y-%m-%d-%H-%M-%S.log")}"""
        )
        self.color_method = color_method
        if not stdout:
            self._clear_default_logger_handlers()
        directory = os.path.dirname(self.filename)
        directory = directory if directory else "."
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.fout = logging.getLogger("NLPAttack-TxtLogger")
        self.fout.setLevel(level=logging.INFO)
        self._init_file_handler()

        self._default_logger.info(f"Logging to text file at path {self.filename}")
        self._num_results = 0
        self._flushed = True

    @property
    def num_results(self) -> int:
        return self._num_results

    def _init_file_handler(self) -> NoReturn:
        """ """
        f_handler = logging.FileHandler(self.filename)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s -\n %(message)s"
        )
        f_handler.setFormatter(f_format)
        self.fout.addHandler(f_handler)

    def __getstate__(self) -> dict:
        # Temporarily save file handle b/c we can't copy it
        state = {
            i: self.__dict__[i]
            for i in self.__dict__
            if i not in ["_default_logger", "fout"]
        }
        return state

    def __setstate__(self, state: dict) -> NoReturn:
        self.__dict__ = state
        if not self.stdout:
            self._clear_default_logger_handlers()
        self.fout = logging.getLogger("NLPAttack-TxtLogger")
        self.fout.setLevel(level=logging.INFO)
        self._init_file_handler()

    def log_attack_result(self, result: AttackResult, **kwargs: Any) -> NoReturn:
        self._num_results += 1
        msg = "\n".join(
            [
                (" Result " + str(self.num_results)).center(110, "-"),
                result.__str__(color_method=self.color_method),
            ]
        )
        self.fout.info(msg)
        if self.stdout:
            self._default_logger.info(msg)
        self._flushed = False

    def log_summary_rows(self, rows: Iterable, title: str, window_id: str) -> NoReturn:
        if self.stdout:
            table_rows = [[title, ""]] + rows
            table = terminaltables.AsciiTable(table_rows)
            self._default_logger.info(table.table)
        else:
            msg = "\n".join([f"{row[0]} {row[1]}" for row in rows])
            self.fout.info(msg)

    def flush(self) -> NoReturn:
        super().flush()
        for h in self.fout.handlers:
            h.flush()
        self._flushed = True

    def close(self) -> NoReturn:
        super().close()
        for h in self.fout.handlers:
            h.close()
            self.fout.removeHandler(h)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "filename",
            "stdout",
            "color_method",
        ]
