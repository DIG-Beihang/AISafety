# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-19
@LastEditTime: 2022-01-05

判断对于分类模型攻击是否成功，
主要基于TextAttack的实现
"""

from typing import Optional, NoReturn, Union, Sequence, List, Tuple

import numpy as np
import torch

from .base import GoalFunction, GoalFunctionResult
from ..strings import (
    color_text,
    process_label_name,
    color_from_label,
    color_from_output,
)
from ..attacked_text import AttackedText


__all__ = [
    "ClassificationGoalFunction",
    "ClassificationGoalFunctionResult",
    "UntargetedClassification",
    "TargetedClassification",
    "InputReduction",
]


class ClassificationGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def _process_model_outputs(
        self, inputs: Sequence, scores: Union[list, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif scores.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            scores = torch.nn.functional.softmax(scores, dim=1)
            if not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
                raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()

    def _goal_function_result_type(self) -> type:
        """Returns the class of this goal function's results."""
        return ClassificationGoalFunctionResult

    def extra_repr_keys(self) -> List[str]:
        return []

    def _get_displayed_output(self, raw_output: torch.Tensor) -> int:
        return int(raw_output.argmax())


class ClassificationGoalFunctionResult(GoalFunctionResult):
    """Represents the result of a classification goal function."""

    @property
    def _processed_output(self) -> Tuple[int, str]:
        """Takes a model output (like `1`) and returns the class labeled output
        (like `positive`), if possible.

        Also returns the associated color.
        """
        output_label = self.raw_output.argmax()
        if self.attacked_text.attack_attrs.get("label_names"):
            output = self.attacked_text.attack_attrs["label_names"][self.output]
            output = process_label_name(output)
            color = color_from_output(output, output_label)
            return output, color
        else:
            color = color_from_label(output_label)
            return output_label, color

    def get_text_color_input(self) -> str:
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        _, color = self._processed_output
        return color

    def get_text_color_perturbed(self) -> str:
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        _, color = self._processed_output
        return color

    def get_colored_output(self, color_method: Optional[str] = None) -> str:
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        output_label = self.raw_output.argmax()
        confidence_score = self.raw_output[output_label]
        if isinstance(confidence_score, torch.Tensor):
            confidence_score = confidence_score.item()
        output, color = self._processed_output
        # concatenate with label and convert confidence score to percent, like '33%'
        output_str = f"{output} ({confidence_score:.0%})"
        return color_text(output_str, color=color, method=color_method)


class UntargetedClassification(ClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(
        self, *args, target_max_score: Optional[float] = None, **kwargs
    ) -> NoReturn:
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output: torch.Tensor, _) -> bool:
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output: torch.Tensor, _) -> float:
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            return 1 - model_output[self.ground_truth_output]


class TargetedClassification(ClassificationGoalFunction):
    """A targeted attack on classification models which attempts to maximize
    the score of the target label.

    Complete when the arget label is the predicted label.
    """

    def __init__(self, *args, target_class: Optional[int] = None, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        assert (
            target_class is not None
        ), "TargetedClassification requires a target class."
        self.target_class = target_class

    def _is_goal_complete(self, model_output: torch.Tensor, _) -> bool:
        return (
            self.target_class == model_output.argmax()
        ) or self.ground_truth_output == self.target_class

    def _get_score(self, model_output: torch.Tensor, _) -> float:
        if self.target_class < 0 or self.target_class >= len(model_output):
            raise ValueError(
                f"target class set to {self.target_class} with {len(model_output)} classes."
            )
        else:
            return model_output[self.target_class]

    def extra_repr_keys(self) -> List[str]:
        if self.maximizable:
            return ["maximizable", "target_class"]
        else:
            return ["target_class"]


class InputReduction(ClassificationGoalFunction):
    """Attempts to reduce the input down to as few words as possible while
    maintaining the same predicted label.
    From Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).
    Pathologies of Neural Models Make Interpretations Difficult.
    https://arxiv.org/abs/1804.07781
    """

    def __init__(self, *args, target_num_words: int = 1, **kwargs) -> NoReturn:
        self.target_num_words = target_num_words
        super().__init__(*args, **kwargs)

    def _is_goal_complete(
        self, model_output: torch.Tensor, attacked_text: AttackedText
    ) -> bool:
        return (
            self.ground_truth_output == model_output.argmax()
            and attacked_text.num_words <= self.target_num_words
        )

    def _should_skip(
        self, model_output: torch.Tensor, attacked_text: AttackedText
    ) -> bool:
        return self.ground_truth_output != model_output.argmax()

    def _get_score(
        self, model_output: torch.Tensor, attacked_text: AttackedText
    ) -> float:
        # Give the lowest score possible to inputs which don't maintain the ground truth label.
        if self.ground_truth_output != model_output.argmax():
            return 0

        cur_num_words = attacked_text.num_words
        initial_num_words = self.initial_attacked_text.num_words

        # The main goal is to reduce the number of words (num_words_score)
        # Higher model score for the ground truth label is used as a tiebreaker (model_score)
        num_words_score = max(
            (initial_num_words - cur_num_words) / initial_num_words, 0
        )
        model_score = model_output[self.ground_truth_output]
        return min(num_words_score + model_score / initial_num_words, 1)

    def extra_repr_keys(self) -> List[str]:
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_num_words"]
