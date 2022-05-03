# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-31
@LastEditTime: 2021-12-03

记录、解析对抗攻击参数的类，
主要基于TextAttack的AttackArgs类进行实现
"""

import os
import sys
import time
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import NoReturn, Optional, List, Union

from ...utils.loggers import AttackLogManager
from ...utils.strings import ReprMixin
from ...utils.misc import nlp_log_dir, str2bool


__all__ = [
    "AttackArgs",
]


@dataclass
class AttackArgs(ReprMixin):
    """Attack arguments to be passed to `AttackerEval`.

    Args:
        num_examples:
            The number of examples to attack. :obj:`-1` for entire dataset.
        num_successful_examples:
            The number of successful adversarial examples we want. This is different from :obj:`num_examples`
            as :obj:`num_examples` only cares about attacking `N` samples while :obj:`num_successful_examples` aims to keep attacking
            until we have `N` successful cases.

            .. note::
                If set, this argument overrides `num_examples` argument.
        num_examples_offset:
            The offset index to start at in the dataset.
        attack_n:
            Whether to run attack until total of `N` examples have been attacked (and not skipped).
        shuffle:
            If :obj:`True`, we randomly shuffle the dataset before attacking. However, this avoids actually shuffling
            the dataset internally and opts for shuffling the list of indices of examples we want to attack. This means
            :obj:`shuffle` can now be used with checkpoint saving.
        query_budget:
            The maximum number of model queries allowed per example attacked.
            If not set, we use the query budget set in the :class:`GoalFunction` object (which by default is :obj:`float("inf")`).

            .. note::
                Setting this overwrites the query budget set in :class:`GoalFunction` object.
        checkpoint_interval:
            If set, checkpoint will be saved after attacking every `N` examples. If :obj:`None` is passed, no checkpoints will be saved.
        checkpoint_dir:
            The directory to save checkpoint files.
        random_seed:
            Random seed for reproducibility.
        parallel:
            If :obj:`True`, run attack using multiple CPUs/GPUs.
        num_workers_per_device:
            Number of worker processes to run per device in parallel mode (i.e. :obj:`parallel=True`). For example, if you are using GPUs and :obj:`num_workers_per_device=2`,
            then 2 processes will be running in each GPU.
        log_to_txt:
            If set, save attack logs as a `.txt` file to the directory specified by this argument.
            If the last part of the provided path ends with `.txt` extension, it is assumed to the desired path of the log file.
        log_to_csv:
            If set, save attack logs as a CSV file to the directory specified by this argument.
            If the last part of the provided path ends with `.csv` extension, it is assumed to the desired path of the log file.
        csv_coloring_style:
            Method for choosing how to mark perturbed parts of the text. Options are :obj:`"file"`, :obj:`"plain"`, and :obj:`"html"`.
            :obj:`"file"` wraps perturbed parts with double brackets :obj:`[[ <text> ]]` while :obj:`"plain"` does not mark the text in any way.
        disable_stdout:
            Disable displaying individual attack results to stdout.
        silent:
            Disable all logging (except for errors). This is stronger than :obj:`disable_stdout`.
        enable_advance_metrics:
            Enable calculation and display of optional advance post-hoc metrics like perplexity, grammar errors, etc.
    """

    __name__ = "AttackArgs"

    num_examples: int = 20
    num_successful_examples: Optional[int] = None
    num_examples_offset: int = 0
    attack_n: bool = False
    shuffle: bool = False
    query_budget: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    random_seed: int = 42
    parallel: bool = False
    num_workers_per_device: int = 1
    log_to_txt: Optional[Union[bool, str]] = True
    log_to_csv: Optional[Union[bool, str]] = True
    log_adv_gen: bool = True
    csv_coloring_style: str = "plain"
    disable_stdout: bool = True
    silent: bool = False
    enable_advance_metrics: bool = False

    def __post_init__(self) -> NoReturn:
        if self.num_successful_examples:
            self.num_examples = None
        if self.num_examples:
            assert (
                self.num_examples >= 0 or self.num_examples == -1
            ), "`num_examples` 必须大于等于0或者等于-1(整个数据集样本数)."
        if self.num_successful_examples:
            assert (
                self.num_successful_examples >= 0
            ), "`num_successful_examples` 必须大于等于0."

        if self.query_budget:
            assert self.query_budget > 0, "`query_budget` 必须大于等于0."

        if self.checkpoint_interval:
            assert self.checkpoint_interval > 0, "`checkpoint_interval` 必须大于等于0."

        assert self.num_workers_per_device > 0, "`num_workers_per_device` 必须大于等于0."

    @classmethod
    def _add_parser_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Add listed args to command line parser."""
        default_obj = cls()
        num_ex_group = parser.add_mutually_exclusive_group(required=False)
        num_ex_group.add_argument(
            "--num-examples",
            "-n",
            type=int,
            default=default_obj.num_examples,
            help="测试样本数目, -1表示整个数据集样本数.",
        )
        num_ex_group.add_argument(
            "--num-successful-examples",
            type=int,
            default=default_obj.num_successful_examples,
            help="对抗攻击成功次数阈值, 达到此阈值即停止测试.",
        )
        parser.add_argument(
            "--num-examples-offset",
            "-o",
            type=int,
            required=False,
            default=default_obj.num_examples_offset,
            help="起始样本编号.",
        )
        parser.add_argument(
            "--query-budget",
            "-q",
            type=int,
            default=default_obj.query_budget,
            help="单个样本对抗攻击次数上限.",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            default=default_obj.shuffle,
            help="是否在对抗攻击前随机打乱数据集顺序.",
        )
        parser.add_argument(
            "--attack-n",
            action="store_true",
            default=default_obj.attack_n,
            help="是否在攻击 (不包含模型本来预测错误而被跳过的样本) 了 `num-examples` 次之后停止.",
        )
        parser.add_argument(
            "--checkpoint-dir",
            required=False,
            type=str,
            default=default_obj.checkpoint_dir,
            help="保存 checkpoint 的目录. 此参数暂未使用.",
        )
        parser.add_argument(
            "--checkpoint-interval",
            required=False,
            type=int,
            default=default_obj.checkpoint_interval,
            help="每攻击 N 个样本保存一个 checkpoint, 不设置则不保存. 此参数暂未使用.",
        )
        parser.add_argument(
            "--random-seed",
            default=default_obj.random_seed,
            type=int,
            help="随机数种子.",
        )
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=default_obj.parallel,
            help="在所有GPU上并行执行.",
        )
        parser.add_argument(
            "--num-workers-per-device",
            default=default_obj.num_workers_per_device,
            type=int,
            help="每个设备上worker数目.",
        )
        parser.add_argument(
            "--log-to-txt",
            nargs="?",
            default=default_obj.log_to_txt,
            const="",
            type=str,
            help="文本日志保存地址.",
        )
        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=default_obj.log_to_csv,
            const="",
            type=str,
            help="CSV日志保存地址.",
        )
        parser.add_argument(
            "--log-adv-gen",
            nargs="?",
            default=default_obj.log_adv_gen,
            type=str2bool,
            help="是否记录生成的对抗样本.",
        )
        parser.add_argument(
            "--csv-coloring-style",
            default=default_obj.csv_coloring_style,
            type=str,
            help='Method for choosing how to mark perturbed parts of the text in CSV logs. Options are "file" and "plain". '
            '"file" wraps text with double brackets `[[ <text> ]]` while "plain" does not mark any text.',
        )
        parser.add_argument(
            "--disable-stdout",
            action="store_true",
            default=default_obj.disable_stdout,
            help="关闭stdout输出.",
        )
        parser.add_argument(
            "--silent",
            action="store_true",
            default=default_obj.silent,
            help="关闭所有日志输出.",
        )
        parser.add_argument(
            "--enable-advance-metrics",
            action="store_true",
            default=default_obj.enable_advance_metrics,
            help="Enable calculation and display of optional advance post-hoc metrics like perplexity, USE distance, etc.",
        )

        return parser

    @classmethod
    def create_loggers_from_args(cls, args: "AttackArgs") -> AttackLogManager:
        """ """
        # Create logger
        attack_log_manager = AttackLogManager()

        # Get current time for file naming
        timestamp = time.strftime("%Y-%m-%d-%H-%M")

        if args.log_adv_gen:
            attack_log_manager.add_adv_gen_csv()

        # if '--log-to-txt' specified with arguments
        if args.log_to_txt:
            if isinstance(args.log_to_txt, bool):
                txt_file_path = os.path.join(nlp_log_dir, f"{timestamp}-log.txt")
            elif args.log_to_txt.lower().endswith(".txt"):
                txt_file_path = args.log_to_txt
            else:
                txt_file_path = os.path.join(args.log_to_txt, f"{timestamp}-log.txt")

            dir_path = os.path.dirname(txt_file_path)
            dir_path = dir_path if dir_path else "."
            if not os.path.exists(dir_path):
                os.makedirs(os.path.dirname(txt_file_path))

            color_method = (
                None if args.csv_coloring_style == "plain" else args.csv_coloring_style
            )
            attack_log_manager.add_output_file(txt_file_path, color_method)

        # if '--log-to-csv' specified with arguments
        if args.log_to_csv:
            if isinstance(args.log_to_csv, bool):
                csv_file_path = os.path.join(nlp_log_dir, f"{timestamp}-log.csv")
            elif args.log_to_csv.lower().endswith(".csv"):
                csv_file_path = args.log_to_csv
            else:
                csv_file_path = os.path.join(args.log_to_csv, f"{timestamp}-log.csv")

            dir_path = os.path.dirname(csv_file_path)
            dir_path = dir_path if dir_path else "."
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            color_method = (
                None if args.csv_coloring_style == "plain" else args.csv_coloring_style
            )
            attack_log_manager.add_output_csv(csv_file_path, color_method)

        # Stdout
        if not args.disable_stdout and not sys.stdout.isatty():
            attack_log_manager.disable_color()
        elif not args.disable_stdout:
            attack_log_manager.enable_stdout()

        return attack_log_manager

    def extra_repr_keys(self) -> List[str]:
        """ """
        return list(self.__dict__.keys())
