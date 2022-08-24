# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-31
@LastEditTime: 2022-03-19

日志管理，多个日志统一管理，
主要基于TextAttack的AttackLogManager进行实现
"""

from typing import NoReturn, Iterable, Optional, List, Union

import numpy as np

from .csv_logger import CSVLogger
from .txt_logger import TxtLogger
from .adv_gen_logger import AdvGenLogger
from ..metrics import (  # noqa: F401
    EditDistanceMetric,
    NumQueriesMetric,
    Perplexity,
    SuccessRate,
    UniversalSentenceEncoderMetric,
    WordsPerturbedMetric,
)
from EvalBox.Attack.TextAttack.attack_result import (  # noqa: F401
    AttackResult,
    FailedAttackResult,
    SkippedAttackResult,
)
from utils.strings import (
    ReprMixin,
    normalize_language,
    LANGUAGE,
    get_str_justify_len,
)


__all__ = [
    "AttackLogManager",
]


class AttackLogManager(ReprMixin):
    """Logs the results of an attack to all attached loggers."""

    __name__ = "AttackLogManager"

    def __init__(
        self,
        language: Union[str, LANGUAGE] = "zh",
        robust_threshold: float = 0.5,
        enable_advance_metrics: bool = False,
        attack_args: Optional[dict] = None,
    ) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        self._robust_threshold = robust_threshold
        self._attack_args = attack_args
        self.loggers = []
        self.results = []
        self.enable_advance_metrics = enable_advance_metrics
        self._algorithm = None
        self._victim_model = None
        self._dataset = None

    def set_language(self, language: Union[str, LANGUAGE]) -> NoReturn:
        """ """
        self._language = normalize_language(language)
        for logger in self.loggers:
            logger.language = self._language

    def enable_stdout(self) -> NoReturn:
        """ """
        if len(self.loggers) == 0:
            self.loggers.append(TxtLogger(stdout=True))
        else:
            self.loggers[0]._init_stream_handler()

    def disable_color(self) -> NoReturn:
        """ """
        if len(self.loggers) == 0:
            self.loggers.append(TxtLogger(stdout=True))
        for logger in self.loggers:
            logger.color_method = "file"

    def add_output_file(self, filename: str, color_method: str) -> NoReturn:
        self.loggers.append(
            TxtLogger(
                language=self._language,
                filename=filename,
                color_method=color_method,
                stdout=False,
            )
        )

    def add_output_csv(self, filename: str, color_method: str) -> NoReturn:
        self.loggers.append(
            CSVLogger(
                language=self._language,
                filename=filename,
                color_method=color_method,
                stdout=False,
            )
        )

    def add_adv_gen_csv(
        self, filename: Optional[str] = None, color_method: Optional[str] = None
    ) -> NoReturn:
        self.loggers.append(AdvGenLogger( stdout=False))

    def log_result(
        self, result: AttackResult, attack_eval: "AttackEval"  # noqa: F821
    ) -> NoReturn:
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result, attack_eval=attack_eval)

    def log_results(
        self, results: Iterable[AttackResult], attack_eval: "AttackEval"  # noqa: F821
    ) -> NoReturn:
        """Logs an iterable of ``AttackResult`` objects on each of `self.loggers`."""
        for result in results:
            self.log_result(result, attack_eval=attack_eval)
        self.log_summary()

    def log_summary_rows(self, rows: Iterable, title: str, window_id: str) -> NoReturn:
        """ """
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self) -> NoReturn:
        """ """
        for logger in self.loggers:
            logger.log_sep()

    def flush(self) -> NoReturn:
        """ """
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name: str, model_name: str) -> NoReturn:
        """ """
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_summary(self) -> NoReturn:
        """ """
        # record the attack args
        if self._attack_args is not None and len(self._attack_args) > 0:
            max_len = (
                max(len(key.encode("GBK")) for key in self._attack_args.keys()) + 2
            )
            args_rows = [
                [(k + ":").ljust(get_str_justify_len(k + ":", max_len)), v]
                for k, v in self._attack_args.items()
            ]
            if self._language == LANGUAGE.ENGLISH:
                self.log_summary_rows(args_rows, "Attack Args", "attack_args")
            elif self._language == LANGUAGE.CHINESE:
                self.log_summary_rows(args_rows, "对抗攻击参数", "attack_args")

        total_attacks = len(self.results)
        if total_attacks == 0:
            return
        # Count things about attacks.
        all_num_words = np.zeros(len(self.results))
        perturbed_word_percentages = np.zeros(len(self.results))
        num_words_changed_until_success = np.zeros(
            2**16
        )  # @ TODO: be smarter about this
        failed_attacks = 0
        skipped_attacks = 0
        successful_attacks = 0
        max_words_changed = 0
        for i, result in enumerate(self.results):
            all_num_words[i] = len(result.original_result.attacked_text.words)
            if isinstance(result, FailedAttackResult):
                failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                skipped_attacks += 1
                continue
            else:
                successful_attacks += 1
            num_words_changed = result.original_result.attacked_text.words_diff_num(
                result.perturbed_result.attacked_text
            )
            num_words_changed_until_success[num_words_changed - 1] += 1
            max_words_changed = max(
                max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_percentages[i] = perturbed_word_percentage

        # Original classifier success rate on these samples.
        original_accuracy = (
            f"{(total_attacks - skipped_attacks) * 100.0 / (total_attacks):.2f}%"
        )

        # New classifier success rate on these samples.
        accuracy_under_attack = f"{(failed_attacks) * 100.0 / (total_attacks):.2f}%"

        # Attack success rate.
        if successful_attacks + failed_attacks == 0:
            _attack_success_rate = 0
        else:
            _attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        attack_success_rate = f"{_attack_success_rate:.2f}%"

        perturbed_word_percentages = perturbed_word_percentages[
            perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = f"{perturbed_word_percentages.mean():.2f}%"
        average_num_words = f"{all_num_words.mean():.2f}"

        if self._language == LANGUAGE.ENGLISH:
            summary_table_rows = [
                ["Attack method:", self._algorithm or "N/A"],
                ["Victim Model:", self._victim_model or "N/A"],
                ["Attack Dataset:", self._dataset or "N/A"],
                ["Number of successful attacks:", str(successful_attacks)],
                ["Number of failed attacks:", str(failed_attacks)],
                ["Number of skipped attacks:", str(skipped_attacks)],
                ["Original accuracy:", original_accuracy],
                ["Accuracy under attack:", accuracy_under_attack],
                ["Attack success rate:", attack_success_rate],
                ["Average perturbed word %:", average_perc_words_perturbed],
                ["Average num. words per input:", average_num_words],
            ]
        elif self._language == LANGUAGE.CHINESE:
            summary_table_rows = [
                ["评测方法:", self._algorithm or "N/A"],
                ["被评测模型:", self._victim_model or "N/A"],
                ["对抗样本来源数据集:", self._dataset or "N/A"],
                ["对抗攻击成功次数:", str(successful_attacks)],
                ["对抗攻击失败次数:", str(failed_attacks)],
                ["跳过对抗攻击次数:", str(skipped_attacks)],
                ["被评测模型原始预测准确率:", original_accuracy],
                ["被攻击后预测准确率:", accuracy_under_attack],
                ["对抗攻击成功率:", attack_success_rate],
                ["对抗样本平均扰动 %:", average_perc_words_perturbed],
                ["对抗样本平均长度（词的数目）:", average_num_words],
            ]

        num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        avg_num_queries = f"{num_queries.mean():.2f}"
        if self._language == LANGUAGE.ENGLISH:
            summary_table_rows.append(["Avg num queries:", avg_num_queries])
        elif self._language == LANGUAGE.CHINESE:
            summary_table_rows.append(["单个样本平均对抗攻击次数:", avg_num_queries])

        # quartiles of num_queries
        quartiles_num_queries = np.percentile(num_queries, [25, 50, 75])
        if self._language == LANGUAGE.ENGLISH:
            summary_table_rows.append(
                [
                    "Quartiles (25%, 50%, 75%) num queries:",
                    ", ".join(f"{n:.3f}" for n in quartiles_num_queries),
                ]
            )
        elif self._language == LANGUAGE.CHINESE:
            summary_table_rows.append(
                [
                    "单个样本对抗攻击次数的四分位数 (25%, 50%, 75%):",
                    ", ".join(f"{n:.3f}" for n in quartiles_num_queries),
                ]
            )

        max_len = max(len(r[0].encode("GBK")) for r in summary_table_rows) + 1
        summary_table_rows = [
            [r[0].ljust(get_str_justify_len(r[0], max_len)), r[1]]
            for r in summary_table_rows
        ]

        if self._language == LANGUAGE.ENGLISH:
            is_robust = (
                " not " if _attack_success_rate > 100 * self._robust_threshold else " "
            )
            summary_table_rows.append(
                [
                    f"Assessed by adv. attack recipe `{self._algorithm}`, model `{self._victim_model}` is{is_robust}robust",
                    "",
                ]
            )
        elif self._language == LANGUAGE.CHINESE:
            is_robust = (
                "不" if _attack_success_rate > 100 * self._robust_threshold else ""
            )
            summary_table_rows.append(
                [
                    f"在对抗攻击方法 `{self._algorithm}` 下，模型 `{self._victim_model}` {is_robust}鲁棒",
                    "",
                ]
            )

        # if self.enable_advance_metrics:
        #     perplexity_stats = Perplexity().calculate(self.results)
        #     use_stats = UniversalSentenceEncoderMetric().calculate(self.results)

        #     summary_table_rows.append(
        #         [
        #             "Average Original Perplexity:",
        #             perplexity_stats["avg_original_perplexity"],
        #         ]
        #     )
        #     summary_table_rows.append(
        #         [
        #             "Average Attack Perplexity:",
        #             perplexity_stats["avg_attack_perplexity"],
        #         ]
        #     )
        #     summary_table_rows.append(
        #         ["Average Attack USE Score:", use_stats["avg_attack_use_score"]]
        #     )

        if self._language == LANGUAGE.ENGLISH:
            self.log_summary_rows(
                summary_table_rows, "Attack Results", "attack_results_summary"
            )
        elif self._language == LANGUAGE.CHINESE:
            self.log_summary_rows(
                summary_table_rows, "对抗攻击结果", "attack_results_summary"
            )

        # Show histogram of words changed.
        numbins = max(max_words_changed, 10)
        for logger in self.loggers:
            logger.log_hist(
                num_words_changed_until_success[:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )

    def extra_repr_keys(self) -> List[str]:
        return [
            "loggers",
        ]
