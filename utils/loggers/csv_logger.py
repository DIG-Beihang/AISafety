# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-31
@LastEditTime: 2021-11-11

CSV文件日志
"""

import os
import time
import csv
from typing import NoReturn, List, Optional, Any

import pandas as pd

from .base import AttackLogger
from ..attacked_text import AttackedText
from ..misc import dict_to_str, nlp_log_dir
from ...EvalBox.Attack.attack_result import AttackResult


__all__ = [
    "CSVLogger",
]


class CSVLogger(AttackLogger):
    """Logs attack results to a CSV."""

    __name__ = "CSVLogger"

    def __init__(
        self,
        filename: Optional[str] = None,
        stdout: bool = False,
        color_method: str = "file",
    ) -> NoReturn:
        """ """
        super().__init__()
        self.filename = filename or os.path.join(
            nlp_log_dir, f"""{time.strftime("CSVLogger-%Y-%m-%d-%H-%M-%S.csv")}"""
        )
        self._default_logger.info(f"Logging to CSV at path {self.filename}")
        self.stdout = stdout
        if not self.stdout:
            self._clear_default_logger_handlers()
        self.color_method = color_method
        self.df = pd.DataFrame()
        self._flushed = True

    def log_attack_result(self, result: AttackResult, **kwargs: Any) -> NoReturn:
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False
        if self.stdout:
            self._default_logger.info(dict_to_str(row))

    def flush(self) -> NoReturn:
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self) -> NoReturn:
        super().close()

    def __del__(self) -> NoReturn:
        if not self._flushed:
            self._default_logger.warning("CSVLogger exiting without calling flush().")

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "filename",
            "stdout",
            "color_method",
        ]
