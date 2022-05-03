# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-11-11
@LastEditTime: 2021-11-11

长期记录生成的攻击样本
"""

import os
import csv
from typing import NoReturn, List, Optional

import pandas as pd

from .base import AttackLogger
from ..misc import dict_to_str, nlp_log_dir
from ...EvalBox.Attack.attack_result import AttackResult
from ...EvalBox.Attack.attack_result import FailedAttackResult, SkippedAttackResult


__all__ = [
    "AdvGenLogger",
]


class AdvGenLogger(AttackLogger):
    """Logs attack results to a CSV."""

    __name__ = "AdvGenLogger"

    def __init__(
        self,
        filename: Optional[str] = None,
        stdout: bool = False,
        color_method: str = "file",
    ) -> NoReturn:
        """ """
        super().__init__()
        self.filename = filename or os.path.join(
            nlp_log_dir.replace("log", "data"), "adv.csv"
        )
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write(
                    "text,label,dataset,algorithm,victim_model,adv_text,adv_output\n"
                )
        self.cols = (
            "text,label,dataset,algorithm,victim_model,adv_text,adv_output".split(",")
        )
        self._default_logger.info(f"Logging to CSV at path {self.filename}")
        self.stdout = stdout
        if not self.stdout:
            self._clear_default_logger_handlers()
        self.df = pd.DataFrame(columns=self.cols)
        self.df_existing = pd.read_csv(self.filename)
        n_existing = len(self.df_existing)
        self.df_existing = self.df_existing.drop_duplicates().reset_index(drop=True)
        if n_existing > len(self.df_existing):
            self.df_existing.to_csv(
                self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False
            )
        self._flushed = True

    def log_attack_result(
        self, result: AttackResult, attack_eval: "AttackEval"  # noqa: F821
    ) -> NoReturn:
        """ """
        if isinstance(
            result,
            (
                FailedAttackResult,
                SkippedAttackResult,
            ),
        ):
            return
        original_text = result.original_text()
        perturbed_text = result.perturbed_text()
        row = {
            "text": original_text,
            "label": result.original_result.ground_truth_output,
            "dataset": attack_eval.dataset.__class__.__name__,
            "algorithm": attack_eval.attack.__class__.__name__,
            "victim_model": attack_eval.attack.model.__class__.__name__,
            "adv_text": perturbed_text,
            "adv_output": result.perturbed_result.output,
        }
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False
        if self.stdout:
            self._default_logger.info(dict_to_str(row))

    def flush(self) -> NoReturn:
        n_existing = len(self.df_existing)
        self.df = (
            pd.concat([self.df_existing, self.df])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.df = self.df.iloc[n_existing:].reset_index(drop=True)
        self.df.to_csv(
            self.filename,
            quoting=csv.QUOTE_NONNUMERIC,
            index=False,
            mode="a",
            header=False,
        )
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
