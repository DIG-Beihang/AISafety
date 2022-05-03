# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2022-04-17

文章Natural Language Adversarial Attack and Defense in Word Level中提出的遗传算法

注意！中文因为切词，可能进行 crossover 的操作之后，词的数目发生变化，
目前暂时把搜索范围截断为两个句子词数目较小的句子的词数目。

"""

from typing import NoReturn, List, Tuple

import numpy as np

from .genetic_algorithm_base import GeneticAlgorithmBase
from .population_based_search import PopulationMember
from ..attacked_text import AttackedText
from ..goal_functions import GoalFunctionResult
from ..misc import DEFAULTS


__all__ = [
    "ImprovedGeneticAlgorithm",
]


class ImprovedGeneticAlgorithm(GeneticAlgorithmBase):
    """ """

    __name__ = "ImprovedGeneticAlgorithm"

    def __init__(
        self,
        pop_size: int = 60,
        max_iters: int = 20,
        temp: float = 0.3,
        give_up_if_no_improvement: bool = False,
        post_crossover_check: bool = True,
        max_crossover_retries: int = 20,
        max_replace_times_per_index: int = 5,
    ) -> NoReturn:
        """
        Args:
            pop_size: The population size. Defaults to 20.
            max_iters: The maximum number of iterations to use. Defaults to 50.
            temp: Temperature for softmax function used to normalize probability dist when sampling parents.
                Higher temperature increases the sensitivity to lower probability candidates.
            give_up_if_no_improvement: If True, stop the search early if no candidate that improves the score is found.
            post_crossover_check: If True, check if child produced from crossover step passes the constraints.
            max_crossover_retries: Maximum number of crossover retries if resulting child fails to pass the constraints.
                Applied only when `post_crossover_check` is set to `True`.
                Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
            max_replace_times_per_index: Maximum times words at the same index can be replaced in this algorithm.
        """
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )

        self.max_replace_times_per_index = max_replace_times_per_index

    def _modify_population_member(
        self,
        pop_member: PopulationMember,
        new_text: AttackedText,
        new_result: GoalFunctionResult,
        word_idx: int,
    ) -> PopulationMember:
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_replacements_left` altered appropriately for given `word_idx`"""
        num_replacements_left = np.copy(pop_member.attributes["num_replacements_left"])
        num_replacements_left[word_idx] -= 1
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_replacements_left": num_replacements_left},
        )

    def _get_word_select_prob_weights(self, pop_member: PopulationMember) -> int:
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        return pop_member.attributes["num_replacements_left"]

    def _crossover_operation(
        self,
        pop_member1: PopulationMember,
        pop_member2: PopulationMember,
    ) -> Tuple[AttackedText, dict]:
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and `pop_member2`.

        Args:
            pop_member1: The first population member.
            pop_member2: The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        indices_to_replace = []
        words_to_replace = []
        num_replacements_left = np.copy(pop_member1.attributes["num_replacements_left"])
        # print("num_replacements_left:", num_replacements_left)
        # print("num_replacements_left.shape:", num_replacements_left.shape)

        # To better simulate the reproduction and biological crossover,
        # IGA randomly cut the text from two parents and concat two fragments into a new text
        # rather than randomly choose a word of each position from the two parents.
        end_point = min(
            pop_member1.num_words, pop_member2.num_words, len(num_replacements_left)
        )
        crossover_point = DEFAULTS.RNG.integers(0, end_point)
        # print(f"crossover_point, pop_member1.num_words, pop_member2.num_words: {crossover_point, pop_member1.num_words, pop_member2.num_words}")
        # if pop_member1.num_words != pop_member2.num_words:
        #     print(f"pop_member1: {pop_member1.attacked_text.text}")
        #     print(f"pop_member2: {pop_member2.attacked_text.text}")
        end_point = max(crossover_point, end_point)
        for i in range(crossover_point, end_point):
            indices_to_replace.append(i)
            words_to_replace.append(pop_member2.words[i])
            num_replacements_left[i] = pop_member2.attributes["num_replacements_left"][
                i
            ]

        new_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return new_text, {"num_replacements_left": num_replacements_left}

    def _initialize_population(
        self,
        initial_result: GoalFunctionResult,
        pop_size: int,
    ) -> List[PopulationMember]:
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result: Original text
            pop_size: size of population
        Returns:
            population as `list[PopulationMember]`
        """
        words = initial_result.attacked_text.words
        # For IGA, `num_replacements_left` represents the number of times the word at each index can be modified
        num_replacements_left = np.array(
            [self.max_replace_times_per_index] * len(words)
        )
        population = []

        # IGA initializes the first population by replacing each word by its optimal synonym
        for idx in range(len(words)):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={"num_replacements_left": np.copy(num_replacements_left)},
            )
            pop_member = self._perturb(pop_member, initial_result, index=idx)
            population.append(pop_member)

        return population[:pop_size]

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + ["max_replace_times_per_index"]
