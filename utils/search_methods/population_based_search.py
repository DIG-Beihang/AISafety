# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-09

基于种群的搜索方法的抽象基类，基于TextAttack进行实现
"""

from abc import ABC, abstractmethod
from typing import NoReturn, List, Optional, Any

from .base import SearchMethod
from ..attacked_text import AttackedText
from ..goal_functions import GoalFunctionResult


class PopulationMember:
    """Represent a single member of population."""

    __name__ = "PopulationMember"

    def __init__(
        self,
        attacked_text: AttackedText,
        result: Optional[GoalFunctionResult] = None,
        attributes: dict = {},
        **kwargs: Any
    ) -> NoReturn:
        self.attacked_text = attacked_text
        self.result = result
        self.attributes = attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def score(self) -> float:
        if not self.result:
            raise ValueError(
                "Result must be obtained for PopulationMember to get its score."
            )
        return self.result.score

    @property
    def words(self) -> List[str]:
        return self.attacked_text.words

    @property
    def num_words(self) -> int:
        return self.attacked_text.num_words


class PopulationBasedSearch(SearchMethod, ABC):
    """Abstract base class for population-based search methods.

    Examples include: genetic algorithm, particle swarm optimization
    """

    __name__ = "PopulationBasedSearch"

    def _check_constraints(
        self,
        transformed_text: AttackedText,
        current_text: AttackedText,
        original_text: AttackedText,
    ) -> NoReturn:
        """Check if `transformted_text` still passes the constraints w.r.t. `current_text` and `original_text`.

        This method is required because of a lot population-based methods does their own transformations apart from
        the actual `transformation`. Examples include `crossover` from `GeneticAlgorithmBase` and `move` from `ParticleSwarmOptimization`.

        Args:
            transformed_text: Resulting text after transformation
            current_text: Recent text from which `transformed_text` was produced from.
            original_text: Original text
        Returns
            `True` if constraints satisfied and `False` if otherwise.
        """
        filtered = self.filter_transformations(
            [transformed_text], current_text, original_text=original_text
        )
        return True if filtered else False

    @abstractmethod
    def _perturb(
        self,
        pop_member: PopulationMember,
        original_result: GoalFunctionResult,
        **kwargs: Any
    ) -> bool:
        """Perturb `pop_member` in-place.

        Must be overridden by specific population-based method
        Args:
            pop_member: Population member to perturb
            original_result: Result for original text. Often needed for constraint checking.
        Returns
            `True` if perturbation occured. `False` if not.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_population(
        self, initial_result: GoalFunctionResult, pop_size: int
    ) -> List[PopulationMember]:
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result: Original text
            pop_size: size of population
        Returns:
            population as `list[PopulationMember]`
        """
        raise NotImplementedError
