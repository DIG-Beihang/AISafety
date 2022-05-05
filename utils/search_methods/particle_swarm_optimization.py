# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2022-04-16

文章Word-level Textual Adversarial Attacking as Combinatorial Optimization中提出的遗传算法
"""

import copy
import warnings
from typing import NoReturn, List, Any, Tuple

import numpy as np

from .population_based_search import PopulationBasedSearch, PopulationMember
from ..goal_functions import GoalFunctionResult, GoalFunctionResultStatus
from ..transformations import (
    Transformation,
    transformation_consists_of_word_substitutes,
)
from ..attacked_text import AttackedText
from ..misc import DEFAULTS


__all__ = [
    "ParticleSwarmOptimization",
]


class ParticleSwarmOptimization(PopulationBasedSearch):
    """Attacks a model with word substiutitions using a Particle Swarm
    Optimization (PSO) algorithm. Some key hyper-parameters are setup according
    to the original paper:

    "We adjust PSO on the validation set of SST and set ω_1 as 0.8 and ω_2 as 0.2.
    We set the max velocity of the particles V_{max} to 3, which means the changing
    probability of the particles ranges from 0.047 (sigmoid(-3)) to 0.953 (sigmoid(3))."
    """

    __name__ = "ParticleSwarmOptimization"

    def __init__(
        self,
        pop_size: int = 60,
        max_iters: int = 20,
        post_turn_check: bool = True,
        max_turn_retries: int = 20,
    ) -> NoReturn:
        """
        Args:
            pop_size: The population size. Defaults to 60.
            max_iters: The maximum number of iterations to use. Defaults to 20.
            post_turn_check: If `True`, check if new position reached by moving passes the constraints. Defaults to `True`
            max_turn_retries: Maximum number of movement retries if new position after turning fails to pass the constraints.
                Applied only when `post_movement_check` is set to `True`.
                Setting it to 0 means we immediately take the old position as the new position upon failure.
        """
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.post_turn_check = post_turn_check
        self.max_turn_retries = 20

        self._search_over = False
        self.omega_1 = 0.8
        self.omega_2 = 0.2
        self.c1_origin = 0.8
        self.c2_origin = 0.2
        self.v_max = 3.0

    def _perturb(
        self, pop_member: PopulationMember, original_result: GoalFunctionResult
    ) -> bool:
        """Perturb `pop_member` in-place.

        Replaces a word at a random in `pop_member` with replacement word that maximizes increase in score.

        Args:
            pop_member: The population member being perturbed.
            original_result: Result of original sample being attacked
        Returns:
            `True` if perturbation occured. `False` if not.
        """
        # TODO: Below is very slow and is the main cause behind memory build up + slowness
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member.result, original_result
        )
        random_result = DEFAULTS.RNG.choice(best_neighbors, 1, p=prob_list)[0]

        if random_result == pop_member.result:
            return False
        else:
            pop_member.attacked_text = random_result.attacked_text
            pop_member.result = random_result
            return True

    def _equal(self, a: Any, b: Any) -> float:
        return -self.v_max if a == b else self.v_max

    def _turn(
        self,
        source_text: PopulationMember,
        target_text: PopulationMember,
        prob: np.ndarray,
        original_text: AttackedText,
    ) -> PopulationMember:
        """
        Based on given probabilities, "move" to `target_text` from `source_text`
        Args:
            source_text: Text we start from.
            target_text: Text we want to move to.
            prob: Turn probability for each word.
            original_text: Original text for constraint check if `self.post_turn_check=True`.
        Returns:
            New `PopulationMember` that we moved to (or if we fail to move, same as `source_text`)
        """
        # assert len(source_text.words) == len(
        #     target_text.words
        # ), "Word length mismatch for turn operation."
        if len(source_text.words) != len(target_text.words):
            warnings.warn("Word length mismatch for turn operation.")
        # assert len(source_text.words) == len(
        #     prob
        # ), "Length mismatch for words and probability list."
        # len_x = len(source_text.words)
        if len(source_text.words) != len(prob):
            warnings.warn("Length mismatch for words and probability list.")

        len_x = min(len(source_text.words), len(target_text.words), len(prob))

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_turn_retries + 1:
            indices_to_replace = []
            words_to_replace = []
            for i in range(len_x):
                if DEFAULTS.RNG.uniform() < prob[i]:
                    indices_to_replace.append(i)
                    words_to_replace.append(target_text.words[i])
            new_text = source_text.attacked_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            indices_to_replace = set(indices_to_replace)
            new_text.attack_attrs["modified_indices"] = (
                source_text.attacked_text.attack_attrs["modified_indices"]
                - indices_to_replace
            ) | (
                target_text.attacked_text.attack_attrs["modified_indices"]
                & indices_to_replace
            )
            if "last_transformation" in source_text.attacked_text.attack_attrs:
                new_text.attack_attrs[
                    "last_transformation"
                ] = source_text.attacked_text.attack_attrs["last_transformation"]

            if not self.post_turn_check or (new_text.words == source_text.words):
                break

            if "last_transformation" in new_text.attack_attrs:
                passed_constraints = self._check_constraints(
                    new_text, source_text.attacked_text, original_text=original_text
                )
            else:
                passed_constraints = True

            if passed_constraints:
                break

            num_tries += 1

        if self.post_turn_check and not passed_constraints:
            # If we cannot find a turn that passes the constraints, we do not move.
            return source_text
        else:
            return PopulationMember(new_text)

    def _get_best_neighbors(
        self,
        current_result: GoalFunctionResult,
        original_result: GoalFunctionResult,
    ) -> Tuple[List[GoalFunctionResult], List[float]]:
        """For given current text, find its neighboring texts that yields
        maximum improvement (in goal function score) for each word.

        Args:
            current_result: `GoalFunctionResult` of current text
            original_result: `GoalFunctionResult` of original text.
        Returns:
            best_neighbors: Best neighboring text for each word
            prob_list: discrete probablity distribution for sampling a neighbor from `best_neighbors`
        """
        current_text = current_result.attacked_text
        neighbors_list = [[] for _ in range(len(current_text.words))]
        transformed_texts = self.get_transformations(
            current_text, original_text=original_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            if diff_idx < len(neighbors_list):
                neighbors_list[diff_idx].append(transformed_text)

        best_neighbors = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                best_neighbors.append(current_result)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(
                neighbors_list[i]
            )
            if not len(neighbor_results):
                best_neighbors.append(current_result)
                score_list.append(0)
            else:
                neighbor_scores = np.array([r.score for r in neighbor_results])
                score_diff = neighbor_scores - current_result.score
                best_idx = np.argmax(neighbor_scores)
                best_neighbors.append(neighbor_results[best_idx])
                score_list.append(score_diff[best_idx])

        prob_list = normalize(score_list)

        return best_neighbors, prob_list

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
        best_neighbors, prob_list = self._get_best_neighbors(
            initial_result, initial_result
        )
        population = []
        for _ in range(pop_size):
            # Mutation step
            random_result = DEFAULTS.RNG.choice(best_neighbors, 1, p=prob_list)[0]
            population.append(
                PopulationMember(random_result.attacked_text, random_result)
            )
        return population

    def perform_search(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        """ """
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        # Initialize  up velocities of each word for each population
        v_init = DEFAULTS.RNG.uniform(-self.v_max, self.v_max, self.pop_size)
        velocities = np.array(
            [
                [v_init[t] for _ in range(initial_result.attacked_text.num_words)]
                for t in range(self.pop_size)
            ]
        )

        global_elite = max(population, key=lambda x: x.score)
        if (
            self._search_over
            or global_elite.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ):
            return global_elite.result

        local_elites = copy.copy(population)

        # start iterations
        for i in range(self.max_iters):
            omega = (self.omega_1 - self.omega_2) * (
                self.max_iters - i
            ) / self.max_iters + self.omega_2
            C1 = self.c1_origin - i / self.max_iters * (self.c1_origin - self.c2_origin)
            C2 = self.c2_origin + i / self.max_iters * (self.c1_origin - self.c2_origin)
            P1 = C1
            P2 = C2

            for k in range(len(population)):
                # calculate the probability of turning each word
                pop_mem_words = population[k].words
                local_elite_words = local_elites[k].words
                assert len(pop_mem_words) == len(
                    local_elite_words
                ), "PSO word length mismatch!"

                for d in range(min(len(velocities[k]), len(pop_mem_words))):
                    velocities[k][d] = omega * velocities[k][d] + (1 - omega) * (
                        self._equal(pop_mem_words[d], local_elite_words[d])
                        + self._equal(pop_mem_words[d], global_elite.words[d])
                    )
                turn_prob = sigmoid(velocities[k])

                if DEFAULTS.RNG.uniform() < P1:
                    # Move towards local elite
                    population[k] = self._turn(
                        local_elites[k],
                        population[k],
                        turn_prob,
                        initial_result.attacked_text,
                    )

                if DEFAULTS.RNG.uniform() < P2:
                    # Move towards global elite
                    population[k] = self._turn(
                        global_elite,
                        population[k],
                        turn_prob,
                        initial_result.attacked_text,
                    )

            # Check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(
                [p.attacked_text for p in population]
            )
            if self._search_over:
                # if `get_goal_results` gets cut short by query budget, resize population
                population = population[: len(pop_results)]
            for k in range(len(pop_results)):
                population[k].result = pop_results[k]

            top_member = max(population, key=lambda x: x.score)
            if (
                self._search_over
                or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Mutation based on the current change rate
            for k in range(len(population)):
                change_ratio = initial_result.attacked_text.words_diff_ratio(
                    population[k].attacked_text
                )
                # Referred from the original source code
                p_change = 1 - 2 * change_ratio
                if DEFAULTS.RNG.uniform() < p_change:
                    self._perturb(population[k], initial_result)

                if self._search_over:
                    break

            # Check if there is any successful attack in the current population
            top_member = max(population, key=lambda x: x.score)
            if (
                self._search_over
                or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Update the elite if the score is increased
            for k in range(len(population)):
                if population[k].score > local_elites[k].score:
                    local_elites[k] = copy.copy(population[k])

            if top_member.score > global_elite.score:
                global_elite = copy.copy(top_member)

        return global_elite.result

    def check_transformation_compatibility(
        self, transformation: Transformation
    ) -> bool:
        """The genetic algorithm is specifically designed for word substitutions."""
        return transformation_consists_of_word_substitutes(transformation)

    @property
    def is_black_box(self) -> bool:
        return True

    def extra_repr_keys(self) -> List[str]:
        return ["pop_size", "max_iters", "post_turn_check", "max_turn_retries"]


def normalize(n: Any) -> np.ndarray:
    n = np.array(n)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    else:
        return n / s


def sigmoid(n: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-n))
