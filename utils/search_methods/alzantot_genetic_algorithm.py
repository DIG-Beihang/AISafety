# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2022-04-17

文章Generating Natural Language Adversarial Examples中提出的遗传算法

"""

from typing import NoReturn, List, Tuple

import numpy as np

from .genetic_algorithm_base import GeneticAlgorithmBase
from .population_based_search import PopulationMember
from ..attacked_text import AttackedText
from ..goal_functions import GoalFunctionResult
from ..misc import DEFAULTS


__all__ = [
    "AlzantotGeneticAlgorithm",
]


class AlzantotGeneticAlgorithm(GeneticAlgorithmBase):
    """ """

    __name__ = "AlzantotGeneticAlgorithm"

    def __init__(
        self,
        pop_size: int = 60,
        max_iters: int = 20,
        temp: float = 0.3,
        give_up_if_no_improvement: bool = False,
        post_crossover_check: bool = True,
        max_crossover_retries: bool = 20,
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
        """
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )

    def _modify_population_member(
        self,
        pop_member: PopulationMember,
        new_text: AttackedText,
        new_result: GoalFunctionResult,
        word_idx: int,
    ) -> PopulationMember:
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_candidate_transformations` altered appropriately for given `word_idx`"""
        num_candidate_transformations = np.copy(
            pop_member.attributes["num_candidate_transformations"]
        )
        num_candidate_transformations[word_idx] = 0
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_candidate_transformations": num_candidate_transformations},
        )

    def _get_word_select_prob_weights(self, pop_member: PopulationMember) -> int:
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        return pop_member.attributes["num_candidate_transformations"]

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
        num_candidate_transformations = np.copy(
            pop_member1.attributes["num_candidate_transformations"]
        )

        for i in range(pop_member1.num_words):
            if DEFAULTS.RNG.uniform() < 0.5:
                indices_to_replace.append(i)
                words_to_replace.append(pop_member2.words[i])
                num_candidate_transformations[i] = pop_member2.attributes[
                    "num_candidate_transformations"
                ][i]

        new_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return (
            new_text,
            {"num_candidate_transformations": num_candidate_transformations},
        )

    def _initialize_population(
        self,
        initial_result: GoalFunctionResult,
        pop_size: int,
    ) -> List[PopulationMember]:
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        words = initial_result.attacked_text.words
        num_candidate_transformations = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            num_candidate_transformations[diff_idx] += 1

        # Just b/c there are no replacements now doesn't mean we never want to select the word for perturbation
        # Therefore, we give small non-zero probability for words with no replacements
        # Epsilon is some small number to approximately assign small probability
        min_num_candidates = np.amin(num_candidate_transformations)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_candidate_transformations)):
            num_candidate_transformations[i] = max(
                num_candidate_transformations[i], epsilon
            )

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={
                    "num_candidate_transformations": np.copy(
                        num_candidate_transformations
                    )
                },
            )
            # Perturb `pop_member` in-place
            pop_member = self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population
