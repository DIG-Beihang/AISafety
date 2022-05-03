# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-23
@LastEditTime: 2022-04-16

对抗攻击的类，
主要基于TextAttack的Attacker类进行实现，同时参考了OpenAttack的类AttackEval

"""

import collections
import logging
import multiprocessing as mp
import os
import queue
import random
import traceback
import warnings
from typing import Optional, Union, List, NoReturn, Deque, Tuple

import numpy as np
import torch
import tqdm

from .attack_result import (  # noqa: F401
    AttackResult,
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from .attack import Attack
from .attack_args import AttackArgs
from ...Datasets.base import NLPDataset
from ...utils.attacked_text import AttackedText
from ...utils.checkpoint import AttackCheckpoint
from ...utils.misc import set_seed
from ...utils.strings import ReprMixin, normalize_language, LANGUAGE


__all__ = [
    "AttackEval",
]


class AttackEval(ReprMixin):
    """
    Class for running attacks on a dataset with specified parameters.

    Example
    -------
    ```python
    from text.EvalBox.Attack.attack_eval import AttackEval
    from text.EvalBox.Attack.attack_args import AttackArgs
    from text.EvalBox.Attack.bae import BAE
    from text.Datasets import SST

    aa = AttackArgs()
    ae = AttackEval(BAE(language="en"), SST(), aa)
    ```
    """

    __name__ = "AttackEval"

    def __init__(
        self,
        attack: Attack,
        dataset: NLPDataset,
        attack_args: Optional[AttackArgs] = None,
    ) -> NoReturn:
        """
        @param {
            attack:
                `Attack` used to actually carry out the attack.
            dataset:
                Dataset to attack.
            attack_args:
                Arguments for attacking the dataset.
            }
        @return: None
        """
        self.attack = attack
        self.dataset = dataset
        assert (
            self.attack.language == self.dataset._language
        ), f"languages of attacker ({self.attack.language}) and dataset ({self.dataset._language}) mismatch!"
        self.language = self.attack.language
        self.attack_args = attack_args or AttackArgs()
        self.attack_log_manager = AttackArgs.create_loggers_from_args(self.attack_args)

        # This is to be set if loading from a checkpoint
        self._checkpoint = None

    def _get_worklist(
        self, start: int, end: int, num_examples: int, shuffle: bool
    ) -> Tuple[Deque, Deque]:
        if end - start < num_examples:
            warnings.warn(
                f"Attempting to attack {num_examples} samples when only {end-start} are available."
            )
        candidates = list(range(start, end))
        if shuffle:
            random.shuffle(candidates)
        worklist = collections.deque(candidates[:num_examples])
        candidates = collections.deque(candidates[num_examples:])
        assert (len(worklist) + len(candidates)) == (end - start)
        return worklist, candidates

    def _attack(
        self, time_out: Optional[float] = None, ignore_errors: bool = False
    ) -> NoReturn:
        """Internal method that carries out attack.

        No parallel processing is involved.

        Args:
            time_out: Timeout in minutes.
            ignore_errors: Whether to ignore errors.

        """
        if torch.cuda.is_available():
            self.attack.cuda_if_possible_()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
            print(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        if not self.attack_args.silent:
            print(self.attack, "\n")

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_skipped = self._checkpoint.num_skipped_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_skipped = 0
            num_successes = 0

        sample_exhaustion_warned = False
        while worklist:
            idx = worklist.popleft()
            try:
                example, ground_truth_output = self.dataset[idx]
            except IndexError:
                continue
            example = AttackedText(self.language, example)
            if self.dataset.label_names is not None:
                example.attack_attrs["label_names"] = self.dataset.label_names
            try:
                result = self.attack.attack(
                    example,
                    ground_truth_output,
                    time_out=time_out,
                    ignore_errors=ignore_errors,
                )
            except Exception as e:
                raise e
            if (
                isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                not isinstance(result, SuccessfulAttackResult)
                and self.attack_args.num_successful_examples
            ):
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    worklist.append(next_sample)
                else:
                    if not sample_exhaustion_warned:
                        warnings.warn("Ran out of samples to attack!")
                        sample_exhaustion_warned = True
            else:
                pbar.update(1)

            self.attack_log_manager.log_result(result, self)
            if not self.attack_args.disable_stdout and not self.attack_args.silent:
                print("\n")
            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        # Enable summary stdout
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()

    def _attack_parallel(
        self, time_out: Optional[float] = None, ignore_errors: bool = False
    ) -> NoReturn:
        """

        Args:
            time_out: Timeout in minutes.
            ignore_errors: If True, errors will be ignored.

        """
        pytorch_multiprocessing_workaround()

        if time_out is None:
            time_out = np.iinfo(np.int32).max

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
            print(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i in worklist:
            try:
                example, ground_truth_output = self.dataset[i]
                example = AttackedText(self.language, example)
                if self.dataset.label_names is not None:
                    example.attack_attrs["label_names"] = self.dataset.label_names
                in_queue.put((i, example, ground_truth_output))
            except IndexError:
                raise IndexError(
                    f"Tried to access element at {i} in dataset of size {len(self.dataset)}."
                )

        # We reserve the first GPU for coordinating workers.
        num_gpus = torch.cuda.device_count()
        num_workers = self.attack_args.num_workers_per_device * num_gpus
        print(f"Running {num_workers} worker(s) on {num_gpus} GPU(s).")

        # Lock for synchronization
        lock = mp.Lock()

        # We move Attacker (and its components) to CPU b/c we don't want models using wrong GPU in worker processes.
        self.attack.cpu_()
        torch.cuda.empty_cache()

        # Start workers.
        worker_pool = torch.multiprocessing.Pool(
            num_workers,
            attack_from_queue,
            (
                self.attack,
                self.attack_args,
                num_gpus,
                mp.Value("i", 1, lock=False),
                lock,
                in_queue,
                out_queue,
            ),
        )

        # Log results asynchronously and update progress bar.
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_skipped = self._checkpoint.num_skipped_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_skipped = 0
            num_successes = 0

        print(f"Worklist size: {len(worklist)}")
        print(f"Worklist candidate size: {len(worklist_candidates)}")

        sample_exhaustion_warned = False
        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
        while worklist:
            idx, result = out_queue.get(block=True)  # TODO: add timeout
            worklist.remove(idx)

            if isinstance(result, tuple) and isinstance(result[0], Exception):
                # TODO: Handle errors if ignore_errors is True.
                print(f'Exception encountered for input "{self.dataset[idx][0]}".')
                error_trace = result[1]
                print(error_trace)
                in_queue.close()
                in_queue.join_thread()
                out_queue.close()
                out_queue.join_thread()
                worker_pool.terminate()
                worker_pool.join()
                return
            elif (
                isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                not isinstance(result, SuccessfulAttackResult)
                and self.attack_args.num_successful_examples
            ):
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    example, ground_truth_output = self.dataset[next_sample]
                    example = AttackedText(self.language, example)
                    if self.dataset.label_names is not None:
                        example.attack_attrs["label_names"] = self.dataset.label_names
                    worklist.append(next_sample)
                    in_queue.put((next_sample, example, ground_truth_output))
                else:
                    if not sample_exhaustion_warned:
                        warnings.warn("Ran out of samples to attack!")
                        sample_exhaustion_warned = True
            else:
                pbar.update()

            self.attack_log_manager.log_result(result, self)
            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        # Send sentinel values to worker processes
        for _ in range(num_workers):
            in_queue.put(("END", "END", "END"))
        worker_pool.close()
        worker_pool.join()

        pbar.close()
        # Enable summary stdout.
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()

    def attack_dataset(
        self, time_out: Optional[float] = None, ignore_errors: bool = False
    ) -> List[AttackResult]:
        """Attack the dataset.

        Args:
            time_out: Timeout in minutes.
            ignore_errors: Whether to ignore errors.

        Returns:
            List of `AttackResult` obtained after attacking the given dataset.
        """
        # if self.attack_args.silent:
        #     logger.setLevel(logging.ERROR)

        if self.attack_args.query_budget:
            self.attack.goal_function.query_budget = self.attack_args.query_budget

        if not self.attack_log_manager:
            self.attack_log_manager = AttackArgs.create_loggers_from_args(
                self.attack_args
            )

        set_seed(self.attack_args.random_seed)
        if self.dataset.shuffled and self.attack_args.checkpoint_interval:
            # Not allowed b/c we cannot recover order of shuffled data
            raise ValueError(
                "Cannot use `--checkpoint-interval` with dataset that has been internally shuffled."
            )

        self.attack_args.num_examples = (
            len(self.dataset)
            if self.attack_args.num_examples == -1
            else self.attack_args.num_examples
        )
        if self.attack_args.parallel:
            if torch.cuda.device_count() == 0:
                raise Exception(
                    "Found no GPU on your system. To run attacks in parallel, GPU is required."
                )
            self._attack_parallel(time_out=time_out, ignore_errors=ignore_errors)
        else:
            self._attack(time_out=time_out, ignore_errors=ignore_errors)

        # if self.attack_args.silent:
        #     logger.setLevel(logging.INFO)

        return self.attack_log_manager.results

    def update_attack_args(self, **kwargs) -> NoReturn:
        """To update any attack args, pass the new argument as keyword argument to this function.

        Examples::

        >>> attacker = #some instance of Attacker
        >>> # To switch to parallel mode and increase checkpoint interval from 100 to 500
        >>> attacker.update_attack_args(parallel=True, checkpoint_interval=500)
        """
        for k in kwargs:
            if hasattr(self.attack_args, k):
                self.attack_args.k = kwargs[k]
            else:
                raise ValueError(f"`AttackArgs` does not have field {k}.")

    @classmethod
    def from_checkpoint(
        cls,
        attack: Attack,
        dataset: NLPDataset,
        checkpoint: Union[str, AttackCheckpoint],
    ) -> "AttackEval":
        """Resume attacking from a saved checkpoint. Attacker and dataset must
        be recovered by the user again, while attack args are loaded from the
        saved checkpoint.

        Args:
            attack:
                Attack object for carrying out the attack.
            dataset:
                Dataset to attack.
            checkpoint：
                Path of saved checkpoint or the actual saved checkpoint.
        """
        if isinstance(checkpoint, str):
            checkpoint = AttackCheckpoint.load(checkpoint)
        attack_eval = cls(attack, dataset, checkpoint.attack_args)
        attack_eval.attack_log_manager = checkpoint.attack_log_manager
        attack_eval._checkpoint = checkpoint
        return attack_eval

    @staticmethod
    def attack_interactive(attack: Attack, language: Union[str, LANGUAGE]) -> NoReturn:
        print(attack, "\n")

        print("Running in interactive mode")
        print("----------------------------")

        while True:
            print(
                f"Enter a sentence in {normalize_language(language).value} to attack or `q` to quit:"
            )
            text = input()

            if text == "q":
                break

            if not text:
                continue

            print("Attacking...")

            example = AttackedText(language, text)
            output = attack.goal_function.get_output(example)
            result = attack.attack(example, output)
            print(result.__str__(color_method="ansi") + "\n")


#
# Helper Methods for multiprocess attacks
#
def pytorch_multiprocessing_workaround() -> NoReturn:
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def set_env_variables(gpu_id: Union[str, int]) -> NoReturn:
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set sharing strategy to file_system to avoid file descriptor leaks
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Only use one GPU, if we have one.
    # For Tensorflow
    # TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For PyTorch
    torch.cuda.set_device(gpu_id)

    # Fix TensorFlow GPU memory growth
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                gpu = gpus[gpu_id]
                tf.config.experimental.set_visible_devices(gpu, "GPU")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except ModuleNotFoundError:
        pass


def attack_from_queue(
    attack: Attack,
    attack_args: AttackArgs,
    num_gpus: int,
    first_to_start: mp.Value,
    lock: mp.Lock,
    in_queue: Deque,
    out_queue: Deque,
) -> NoReturn:
    """ """
    assert isinstance(
        attack, Attack
    ), f"`attack` must be of type `Attack`, but got type `{type(attack).__name__}`."

    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    set_env_variables(gpu_id)
    set_seed(attack_args.random_seed)
    if torch.multiprocessing.current_process()._identity[0] > 1:
        logging.disable()

    attack.cuda_if_possible_()

    # Simple non-synchronized check to see if it's the first process to reach this point.
    # This let us avoid waiting for lock.
    if bool(first_to_start.value):
        # If it's first process to reach this step, we first try to acquire the lock to update the value.
        with lock:
            # Because another process could have changed `first_to_start=False` while we wait, we check again.
            if bool(first_to_start.value):
                first_to_start.value = 0
                if not attack_args.silent:
                    print(attack, "\n")

    while True:
        try:
            i, example, ground_truth_output = in_queue.get(timeout=5)
            if i == "END" and example == "END" and ground_truth_output == "END":
                # End process when sentinel value is received
                break
            else:
                result = attack.attack(example, ground_truth_output)
                out_queue.put((i, result))
        except Exception as e:
            if isinstance(e, queue.Empty):
                continue
            else:
                out_queue.put((i, (e, traceback.format_exc())))
