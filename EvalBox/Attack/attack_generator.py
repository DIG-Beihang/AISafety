# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-21
@LastEditTime: 2022-04-16

文本对抗攻击过程中生成对抗样本的类，
主要基于TextAttack的类Attack以及AttackRecipe进行实现，同时参考了OpenAttack的类Attacker

"""

from abc import ABC
from collections import OrderedDict
from typing import Union, List, NoReturn, Any, Optional
import textwrap
import traceback

import lru
import torch

from ...utils.attacked_text import AttackedText
from ...utils.transformations import Transformation, CompositeTransformation
from ...utils.goal_functions import (
    GoalFunction,
    GoalFunctionResult,
    GoalFunctionResultStatus,
)
from ...utils.constraints import Constraint, PreTransformationConstraint
from ...utils.search_methods import SearchMethod
from ...utils.misc import default_device, hashable, timeout
from ...utils.strings import LANGUAGE, normalize_language
from ...Models.base import NLPVictimModel
from .attack_result import (  # noqa: F401
    AttackResult,
    SuccessfulAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    FailedAttackResult,
)


__all__ = [
    "AdvSampleGenerator",
    "AttackGenerator",
]


class AdvSampleGenerator(object):
    """ """

    __name__ = "AdvSampleGenerator"

    def __init__(
        self,
        constraints: List[Union[Constraint, PreTransformationConstraint]],
        transformation: Transformation,
        transformation_cache_size: int = 2**15,
        constraint_cache_size: int = 2**15,
        language: Union[str, LANGUAGE] = "en",
        verbose: int = 0,
    ) -> NoReturn:
        """
        @param {
            constraints:
                A list of constraints to add to the attack, defining which perturbations are valid.
            transformation:
                The transformation applied at each step of the attack.
            transformation_cache_size:
                The number of items to keep in the transformations cache
            constraint_cache_size:
                The number of items to keep in the constraints cache
            }
        @return: None
        """
        self.language = normalize_language(language)
        self.verbose = verbose
        self.transformation = transformation

        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(
                constraint,
                PreTransformationConstraint,
            ):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

        # Check if we can use transformation cache for our transformation.
        if not self.transformation.deterministic:
            self.use_transformation_cache = False
        elif isinstance(self.transformation, CompositeTransformation):
            self.use_transformation_cache = True
            for t in self.transformation.transformations:
                if not t.deterministic:
                    self.use_transformation_cache = False
                    break
        else:
            self.use_transformation_cache = True
        self.transformation_cache_size = transformation_cache_size
        self.transformation_cache = lru.LRU(transformation_cache_size)

        self.constraint_cache_size = constraint_cache_size
        self.constraints_cache = lru.LRU(constraint_cache_size)

    def clear_cache(self, recursive=True):
        self.constraints_cache.clear()
        if self.use_transformation_cache:
            self.transformation_cache.clear()
        if recursive:
            for constraint in self.constraints:
                if hasattr(constraint, "clear_cache"):
                    constraint.clear_cache()

    def cpu_(self) -> NoReturn:
        """Move any `torch.nn.Module` models that are part of Attack to CPU."""
        visited = set()

        def to_cpu(obj):
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                obj.cpu()
            elif isinstance(
                obj,
                (
                    AdvSampleGenerator,
                    GoalFunction,
                    Transformation,
                    SearchMethod,
                    Constraint,
                    PreTransformationConstraint,
                    NLPVictimModel,
                ),
            ):
                for key in obj.__dict__:
                    s_obj = obj.__dict__[key]
                    if id(s_obj) not in visited:
                        to_cpu(s_obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if id(item) not in visited and isinstance(
                        item, (Transformation, Constraint, PreTransformationConstraint)
                    ):
                        to_cpu(item)

        to_cpu(self)

    def cuda_(self) -> NoReturn:
        """Move any `torch.nn.Module` models that are part of Attack to GPU."""
        visited = set()

        def to_cuda(obj):
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                try:
                    obj.to(default_device)
                except Exception as e:
                    print(f"{obj.__class__.__name__} to cuda failed")
                    raise e
            elif isinstance(
                obj,
                (
                    AdvSampleGenerator,
                    GoalFunction,
                    Transformation,
                    SearchMethod,
                    Constraint,
                    PreTransformationConstraint,
                    NLPVictimModel,
                ),
            ):
                for key in obj.__dict__:
                    s_obj = obj.__dict__[key]
                    if id(s_obj) not in visited:
                        to_cuda(s_obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if id(item) not in visited and isinstance(
                        item, (Transformation, Constraint, PreTransformationConstraint)
                    ):
                        to_cuda(item)

        to_cuda(self)

    def cuda_if_possible_(self) -> bool:
        """ """
        try:
            self.cuda_()
            print(f"{self.__name__} to cuda succeeded")
            return True
        except Exception:
            self.cpu_()
            print(f"{self.__name__} to cuda failded, defaults to cpu")
            print("exceptions are as follows")
            print(traceback.format_exc())
            return False

    def _get_transformations_uncached(
        self,
        current_text: AttackedText,
        original_text: Optional[AttackedText] = None,
        **kwargs: Any,
    ) -> List[AttackedText]:
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformationed ``AttackedText`` through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformationed where each transformation matches the constraints
        """
        transformed_texts = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            # original_text,
            **kwargs,
        )

        return transformed_texts

    def get_transformations(
        self,
        current_text: AttackedText,
        original_text: Optional[AttackedText] = None,
        **kwargs: Any,
    ) -> List[AttackedText]:
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        if not self.transformation:
            raise RuntimeError(
                "Cannot call `get_transformations` without a transformation."
            )

        if self.use_transformation_cache:
            cache_key = tuple([current_text] + sorted(kwargs.items()))
            if hashable(cache_key) and cache_key in self.transformation_cache:
                # promote transformed_text to the top of the LRU cache
                self.transformation_cache[cache_key] = self.transformation_cache[
                    cache_key
                ]
                transformed_texts = list(self.transformation_cache[cache_key])
            else:
                transformed_texts = self._get_transformations_uncached(
                    current_text, original_text, **kwargs
                )
                if hashable(cache_key):
                    self.transformation_cache[cache_key] = tuple(transformed_texts)
        else:
            transformed_texts = self._get_transformations_uncached(
                current_text, original_text, **kwargs
            )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )

    def _filter_transformations_uncached(
        self,
        transformed_texts: List[AttackedText],
        current_text: AttackedText,
        original_text: Optional[AttackedText] = None,
    ) -> List[AttackedText]:
        """Filters a list of potential transformed texts based on
        ``self.constraints``

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        filtered_texts = transformed_texts[:]
        for C in self.constraints:
            if len(filtered_texts) == 0:
                break
            if C.compare_against_original:
                if not original_text:
                    raise ValueError(
                        f"Missing `original_text` argument when constraint {type(C)} is set to compare against `original_text`"
                    )

                filtered_texts = C.check_batch(filtered_texts, original_text)
            else:
                filtered_texts = C.check_batch(filtered_texts, current_text)
        # Default to false for all original transformations.
        for original_transformed_text in transformed_texts:
            self.constraints_cache[(current_text, original_transformed_text)] = False
        # Set unfiltered transformations to True in the cache.
        for filtered_text in filtered_texts:
            self.constraints_cache[(current_text, filtered_text)] = True
        return filtered_texts

    def filter_transformations(
        self,
        transformed_texts: List[AttackedText],
        current_text: AttackedText,
        original_text: AttackedText = None,
    ) -> List[AttackedText]:
        """Filters a list of potential transformed texts based on
        ``self.constraints`` Utilizes an LRU cache to attempt to avoid
        recomputing common transformations.

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        # Remove any occurences of current_text in transformed_texts
        transformed_texts = [
            t for t in transformed_texts if t.text != current_text.text
        ]
        # Populate cache with transformed_texts
        uncached_texts = []
        filtered_texts = []
        for transformed_text in transformed_texts:
            if (current_text, transformed_text) not in self.constraints_cache:
                uncached_texts.append(transformed_text)
            else:
                # promote transformed_text to the top of the LRU cache
                self.constraints_cache[
                    (current_text, transformed_text)
                ] = self.constraints_cache[(current_text, transformed_text)]
                if self.constraints_cache[(current_text, transformed_text)]:
                    filtered_texts.append(transformed_text)
        filtered_texts += self._filter_transformations_uncached(
            uncached_texts, current_text, original_text=original_text
        )
        # Sort transformations to ensure order is preserved between runs
        filtered_texts.sort(key=lambda t: t.text)
        return filtered_texts

    def __repr__(self) -> str:
        """Prints attack parameters in a human-readable string.

        Inspired by the readability of printing PyTorch nn.Modules:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
        """
        main_str = f"{self.__name__}("
        lines = self._generate_repr_lines()
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def _generate_repr_lines(self) -> List[str]:
        """
        Generates a list of strings that can be used to print the class
        """
        lines = []
        # self.transformation
        lines.append(
            textwrap.indent(f"(transformation):  {self.transformation}", " " * 2)
        )
        # self.constraints
        constraints_lines = []
        constraints = self.constraints + self.pre_transformation_constraints
        if len(constraints):
            for i, constraint in enumerate(constraints):
                constraints_lines.append(
                    textwrap.indent(f"({i}): {constraint}", " " * 2)
                )
            constraints_str = textwrap.indent(
                "\n" + "\n".join(constraints_lines), " " * 2
            )
        else:
            constraints_str = "None"
        lines.append(textwrap.indent(f"(constraints): {constraints_str}", " " * 2))
        return lines

    __str__ = __repr__

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["transformation_cache"] = None
        state["constraints_cache"] = None
        return state

    def __setstate__(self, state: dict) -> NoReturn:
        self.__dict__ = state
        self.transformation_cache = lru.LRU(self.transformation_cache_size)
        self.constraints_cache = lru.LRU(self.constraint_cache_size)


class AttackGenerator(AdvSampleGenerator, ABC):
    """An attack generates adversarial examples on text.

    An attack is comprised of a goal function, constraints, transformation, and a search method. Use :meth:`attack` method to attack one sample at a time.

    adopted from TextAttack
    """

    __name__ = "AttackGenerator"

    def __init__(
        self,
        goal_function: GoalFunction,
        constraints: List[Union[Constraint, PreTransformationConstraint]],
        transformation: Transformation,
        search_method: SearchMethod,
        transformation_cache_size: int = 2**15,
        constraint_cache_size: int = 2**15,
        language: Union[str, LANGUAGE] = "en",
        verbose: int = 0,
    ) -> NoReturn:
        """
        @param {
            goal_function:
                A function for determining how well a perturbation is doing at achieving the attack's goal.
            constraints:
                A list of constraints to add to the attack, defining which perturbations are valid.
            transformation:
                The transformation applied at each step of the attack.
            search_method:
                The method for exploring the search space of possible perturbations
            transformation_cache_size:
                The number of items to keep in the transformations cache
            constraint_cache_size:
                The number of items to keep in the constraints cache
            }
        @return: None
        """
        super().__init__(
            constraints=constraints,
            transformation=transformation,
            transformation_cache_size=transformation_cache_size,
            constraint_cache_size=constraint_cache_size,
            language=language,
            verbose=verbose,
        )
        self.goal_function = goal_function
        self.search_method = search_method

        self.is_black_box = (
            getattr(transformation, "is_black_box", True) and search_method.is_black_box
        )

        if not self.search_method.check_transformation_compatibility(
            self.transformation
        ):
            raise ValueError(
                f"SearchMethod {self.search_method} incompatible with transformation {self.transformation}"
            )

        # Give search method access to functions for getting transformations and evaluating them
        self.search_method.get_transformations = self.get_transformations
        # Give search method access to self.goal_function for model query count, etc.
        self.search_method.goal_function = self.goal_function
        # The search method only needs access to the first argument. The second is only used
        # by the attack class when checking whether to skip the sample
        self.search_method.get_goal_results = self.goal_function.get_results

        self.search_method.filter_transformations = self.filter_transformations

    def clear_cache(self, recursive=True):
        self.constraints_cache.clear()
        if self.use_transformation_cache:
            self.transformation_cache.clear()
        if recursive:
            self.goal_function.clear_cache()
            for constraint in self.constraints:
                if hasattr(constraint, "clear_cache"):
                    constraint.clear_cache()

    def _attack(self, initial_result: GoalFunctionResult) -> AttackResult:
        """Calls the ``SearchMethod`` to perturb the ``AttackedText`` stored in
        ``initial_result``.

        Args:
            initial_result: The initial ``GoalFunctionResult`` from which to perturb.

        Returns:
            A ``SuccessfulAttackResult``, ``FailedAttackResult``,
                or ``MaximizedAttackResult``.
        """
        if self.verbose >= 2:
            print("start searching for adversarial examples...")
        final_result = self.search_method(initial_result)
        self.clear_cache()
        if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            return SuccessfulAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.SEARCHING:
            return FailedAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.MAXIMIZING:
            return MaximizedAttackResult(
                initial_result,
                final_result,
            )
        else:
            raise ValueError(f"Unrecognized goal status {final_result.goal_status}")

    def attack(
        self,
        example: Union[str, OrderedDict],
        ground_truth_output: Union[str, int],
        time_out: Optional[float] = None,
        ignore_errors: bool = False,
    ) -> AttackResult:
        """Attack a single example."""
        if time_out is None:
            time_out = float("inf")
        if isinstance(example, (str, OrderedDict)):
            example = AttackedText(self.language, example)

        if self.verbose >= 2:
            print(
                f"start attacking using the example\n{example}\nwith ground_truth {ground_truth_output}\n"
            )

        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."
        goal_function_result, _ = self.goal_function.init_attack_example(
            example, ground_truth_output
        )
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            try:
                with timeout(duration=60 * time_out):
                    result = self._attack(goal_function_result)
            except TimeoutError:
                print("time out")
                result = FailedAttackResult(goal_function_result, goal_function_result)
            except Exception:  # RuntimeError: CUDA out of memory
                self.cpu_()
                try:
                    with timeout(duration=60 * time_out):
                        result = self._attack(goal_function_result)
                except TimeoutError:
                    print("time out")
                    result = FailedAttackResult(
                        goal_function_result, goal_function_result
                    )
                except Exception as e:
                    if ignore_errors:
                        print(e)
                        result = FailedAttackResult(
                            goal_function_result, goal_function_result
                        )
                    else:
                        raise e
            return result

    def _generate_repr_lines(self) -> List[str]:
        """
        Generates a list of strings that can be used to print the class
        """
        lines = []
        lines.append(textwrap.indent(f"(search_method): {self.search_method}", " " * 2))
        # self.goal_function
        lines.append(
            textwrap.indent(f"(goal_function):  {self.goal_function}", " " * 2)
        )
        # self.transformation and self.constraints
        lines.extend(super()._generate_repr_lines())
        # self.is_black_box
        lines.append(textwrap.indent(f"(is_black_box):  {self.is_black_box}", " " * 2))
        return lines

    # @staticmethod
    # @abstractmethod
    # def build(model_wrapper:NLPVictimModel, **kwargs) -> "AttackGenerator":
    #     """
    #     静态方法，可用于直接生成`AttackGenerator`
    #     """
    #     raise NotImplementedError
