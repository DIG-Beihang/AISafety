# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-19
@LastEditTime: 2022-03-19

判断对于模型攻击是否成功的抽象基类，
主要基于TextAttack的实现
"""

from typing import Any, List, NoReturn, Optional
from abc import ABC, abstractmethod
from enum import IntEnum

import lru
import numpy as np
import torch

from ..attacked_text import AttackedText
from ..strings import ReprMixin
from ...Models.base import NLPVictimModel


__all__ = [
    "GoalFunction",
    "GoalFunctionResultStatus",
    "GoalFunctionResult",
]


class GoalFunction(ReprMixin, ABC):
    """
    Evaluates how well a perturbed attacked_text object is achieving a specified goal.
    """

    __name__ = "GoalFunction"

    def __init__(
        self,
        model_wrapper: NLPVictimModel,
        maximizable: bool = False,
        use_cache: bool = True,
        query_budget: float = float("inf"),
        model_batch_size: int = 32,
        model_cache_size: int = 2**20,
    ) -> NoReturn:
        """
        @param {
            model_wrapper: The victim model to attack.
            maximizable:
                Whether the goal function is maximizable, as opposed to a boolean result of success or failure.
            query_budget:
                The maximum number of model queries allowed.
            model_cache_size:
                The maximum number of items to keep in the model results cache at once.
        }
        @return: None
        """
        self.model = model_wrapper
        self.maximizable = maximizable
        self.use_cache = use_cache
        self.query_budget = query_budget
        self.batch_size = model_batch_size
        if self.use_cache:
            self._call_model_cache = lru.LRU(model_cache_size)
        else:
            self._call_model_cache = None

    def clear_cache(self) -> NoReturn:
        if self.use_cache:
            self._call_model_cache.clear()

    def init_attack_example(
        self, attacked_text: AttackedText, ground_truth_output: Any
    ) -> tuple:
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def get_output(self, attacked_text: AttackedText) -> str:
        """Returns output for display based on the result of calling the model."""
        return self._get_displayed_output(self._call_model([attacked_text])[0])

    def get_result(self, attacked_text: AttackedText, **kwargs) -> tuple:
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_terminates = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_terminates

    def get_results(
        self, attacked_text_list: List[AttackedText], check_skip: bool = False
    ) -> tuple:
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def _get_goal_status(
        self, model_output: Any, attacked_text: AttackedText, check_skip: bool = False
    ) -> "GoalFunctionResultStatus":
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    @abstractmethod
    def _is_goal_complete(self, model_output: Any, attacked_text: AttackedText) -> bool:
        raise NotImplementedError

    def _should_skip(self, model_output: Any, attacked_text: AttackedText) -> bool:
        return self._is_goal_complete(model_output, attacked_text)

    @abstractmethod
    def _get_score(self, model_output: Any, attacked_text: AttackedText) -> Any:
        raise NotImplementedError

    def _get_displayed_output(self, raw_output: Any) -> Any:
        return raw_output

    @abstractmethod
    def _goal_function_result_type(self) -> Any:
        """Returns the class of this goal function's results."""
        raise NotImplementedError

    @abstractmethod
    def _process_model_outputs(self, inputs: Any, outputs: Any) -> Any:
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to make sure they have a softmax applied.
        """
        raise NotImplementedError

    def _call_model_uncached(self, attacked_text_list: List[AttackedText]) -> Any:
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        inputs = [at.tokenizer_input for at in attacked_text_list]
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            batch_preds = self.model(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)

    def _call_model(self, attacked_text_list: List[AttackedText]) -> list:
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
        if not self.use_cache:
            return self._call_model_uncached(attacked_text_list)
        else:
            uncached_list = []
            for text in attacked_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    uncached_list.append(text)
            uncached_list = [
                text
                for text in attacked_text_list
                if text not in self._call_model_cache
            ]
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs

    def extra_repr_keys(self) -> List[str]:
        attrs = []
        if self.query_budget < float("inf"):
            attrs.append("query_budget")
        if self.maximizable:
            attrs.append("maximizable")
        return attrs

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.use_cache:
            state["_call_model_cache"] = self._call_model_cache.get_size()
        return state

    def __setstate__(self, state) -> NoReturn:
        self.__dict__ = state
        if self.use_cache:
            self._call_model_cache = lru.LRU(state["_call_model_cache"])


class GoalFunctionResultStatus(IntEnum):
    """ """

    SUCCEEDED = 0
    SEARCHING = 1  # In process of searching for a success
    MAXIMIZING = 2
    SKIPPED = 3


class GoalFunctionResult(ReprMixin, ABC):
    """Represents the result of a goal function evaluating a AttackedText
    object.

    Args:
        attacked_text: The sequence that was evaluated.
        output: The display-friendly output.
        goal_status: The ``GoalFunctionResultStatus`` representing the status of the achievement of the goal.
        score: A score representing how close the model is to achieving its goal.
        num_queries: How many model queries have been used
        ground_truth_output: The ground truth output
    """

    __name__ = "GoalFunctionResult"

    def __init__(
        self,
        attacked_text: AttackedText,
        raw_output: Any,
        output: Any,
        goal_status: GoalFunctionResultStatus,
        score: Any,
        num_queries: int,
        ground_truth_output: Any,
    ) -> NoReturn:
        self.attacked_text = attacked_text
        self.raw_output = raw_output
        self.output = output
        self.score = score
        self.goal_status = goal_status
        self.num_queries = num_queries
        self.ground_truth_output = ground_truth_output

        if isinstance(self.raw_output, torch.Tensor):
            self.raw_output = self.raw_output.numpy()

        if isinstance(self.score, torch.Tensor):
            self.score = self.score.item()

    @abstractmethod
    def get_text_color_input(self) -> str:
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        raise NotImplementedError

    @abstractmethod
    def get_text_color_perturbed(self) -> str:
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        raise NotImplementedError

    @abstractmethod
    def get_colored_output(self, color_method: Optional[callable] = None) -> str:
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        attrs = [
            "goal_status",
            "score",
            "output",
            "ground_truth_output",
        ]
        return attrs
