# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-01
@LastEditTime: 2022-03-19

记录对抗攻击的checkpoints，方便暂停后继续进行
"""

import copy
import os
import pickle
import time
import textwrap
from datetime import datetime
from typing import Sequence, NoReturn, Optional

from .loggers import AttackLogManager
from ..EvalBox.Attack.attack_args import AttackArgs
from ..EvalBox.Attack.attack_result import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)


__all__ = [
    "AttackCheckpoint",
]


class AttackCheckpoint:
    """An object that stores necessary information for saving and loading
    checkpoints.

    Args:
        attack_args:
            Arguments of the original attack
        attack_log_manager:
            Object for storing attack results
        worklist:
            List of examples that will be attacked. Examples are represented by their indicies within the dataset.
        worklist_candidates:
            List of other available examples we can attack. Used to get the next dataset element when `attack_n=True`.
        chkpt_ts:
            epoch timestamp representing when checkpoint was made
    """

    def __init__(
        self,
        attack_args: AttackArgs,
        attack_log_manager: AttackLogManager,
        worklist: Sequence[int],
        worklist_candidates: Sequence[int],
        chkpt_ts: Optional[float] = None,
    ):
        self.attack_args = copy.deepcopy(attack_args)
        self.attack_log_manager = attack_log_manager
        self.worklist = worklist
        self.worklist_candidates = worklist_candidates
        if chkpt_ts:
            self.time = chkpt_ts
        else:
            self.time = time.time()

        self._verify()

    @property
    def results_count(self) -> int:
        """Return number of attacks made so far."""
        return len(self.attack_log_manager.results)

    @property
    def num_skipped_attacks(self) -> int:
        return sum(
            isinstance(r, SkippedAttackResult) for r in self.attack_log_manager.results
        )

    @property
    def num_failed_attacks(self) -> int:
        return sum(
            isinstance(r, FailedAttackResult) for r in self.attack_log_manager.results
        )

    @property
    def num_successful_attacks(self) -> int:
        return sum(
            isinstance(r, SuccessfulAttackResult)
            for r in self.attack_log_manager.results
        )

    @property
    def num_maximized_attacks(self) -> int:
        return sum(
            isinstance(r, MaximizedAttackResult)
            for r in self.attack_log_manager.results
        )

    @property
    def num_remaining_attacks(self) -> int:
        if self.attack_args.attack_n:
            non_skipped_attacks = self.num_successful_attacks + self.num_failed_attacks
            count = self.attack_args.num_examples - non_skipped_attacks
        else:
            count = self.attack_args.num_examples - self.results_count
        return count

    @property
    def dataset_offset(self) -> int:
        """Calculate offset into the dataset to start from."""
        # Original offset + # of results processed so far
        return self.attack_args.num_examples_offset + self.results_count

    @property
    def datetime(self) -> str:
        return datetime.fromtimestamp(self.time).strftime("%Y-%m-%d %H:%M:%S")

    def save(self, quiet: bool = False) -> NoReturn:
        file_name = f"{int(self.time * 1000)}.attack.ckpt"
        os.makedirs(self.attack_args.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.attack_args.checkpoint_dir, file_name)
        if not quiet:
            print("\n\n" + "=" * 125)
            print(
                'Saving checkpoint under "{}" at {} after {} attacks.'.format(
                    path, self.datetime, self.results_count
                )
            )
            print("=" * 125 + "\n")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "AttackCheckpoint":
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        assert isinstance(checkpoint, cls)

        return checkpoint

    def _verify(self) -> NoReturn:
        """Check that the checkpoint has no duplicates and is consistent."""
        assert self.num_remaining_attacks == len(
            self.worklist
        ), "Recorded number of remaining attacks and size of worklist are different."

        results_set = set()
        for result in self.attack_log_manager.results:
            results_set.add(result.original_text)

        assert (
            len(results_set) == self.results_count
        ), "Duplicate `AttackResults` found."

    def __repr__(self) -> str:
        main_str = "AttackCheckpoint("
        lines = []
        lines.append(textwrap.indent(f"(Time):  {self.datetime}", " " * 2))

        args_lines = []
        recipe_set = (
            True
            if "recipe" in self.attack_args.__dict__
            and self.attack_args.__dict__["recipe"]
            else False
        )
        mutually_exclusive_args = ["search", "transformation", "constraints", "recipe"]
        if recipe_set:
            args_lines.append(
                textwrap.indent(
                    f'(recipe): {self.attack_args.__dict__["recipe"]}', " " * 2
                )
            )
        else:
            args_lines.append(
                textwrap.indent(
                    f'(search): {self.attack_args.__dict__["search"]}', " " * 2
                )
            )
            args_lines.append(
                textwrap.indent(
                    f'(transformation): {self.attack_args.__dict__["transformation"]}',
                    " " * 2,
                )
            )
            args_lines.append(
                textwrap.indent(
                    f'(constraints): {self.attack_args.__dict__["constraints"]}',
                    " " * 2,
                )
            )

        for key in self.attack_args.__dict__:
            if key not in mutually_exclusive_args:
                args_lines.append(
                    textwrap.indent(
                        f"({key}): {self.attack_args.__dict__[key]}", " " * 2
                    )
                )

        args_str = textwrap.indent("\n" + "\n".join(args_lines), " " * 2)
        lines.append(textwrap.indent(f"(attack_args):  {args_str}", " " * 2))

        attack_logger_lines = []
        attack_logger_lines.append(
            textwrap.indent(
                f"(Total number of examples to attack): {self.attack_args.num_examples}",
                " " * 2,
            )
        )
        attack_logger_lines.append(
            textwrap.indent(
                f"(Number of attacks performed): {self.results_count}", " " * 2
            )
        )
        attack_logger_lines.append(
            textwrap.indent(
                f"(Number of remaining attacks): {self.num_remaining_attacks}", " " * 2
            )
        )
        breakdown_lines = []
        breakdown_lines.append(
            textwrap.indent(
                f"(Number of successful attacks): {self.num_successful_attacks}",
                " " * 2,
            )
        )
        breakdown_lines.append(
            textwrap.indent(
                f"(Number of failed attacks): {self.num_failed_attacks}", " " * 2
            )
        )
        breakdown_lines.append(
            textwrap.indent(
                f"(Number of maximized attacks): {self.num_maximized_attacks}", " " * 2
            )
        )
        breakdown_lines.append(
            textwrap.indent(
                f"(Number of skipped attacks): {self.num_skipped_attacks}", " " * 2
            )
        )
        breakdown_str = textwrap.indent("\n" + "\n".join(breakdown_lines), " " * 2)
        attack_logger_lines.append(
            textwrap.indent(f"(Latest result breakdown): {breakdown_str}", " " * 2)
        )
        attack_logger_str = textwrap.indent(
            "\n" + "\n".join(attack_logger_lines), " " * 2
        )
        lines.append(
            textwrap.indent(f"(Previous attack summary):  {attack_logger_str}", " " * 2)
        )

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    __str__ = __repr__
