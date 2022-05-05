# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-08
@LastEditTime: 2021-09-13

An attack that maintinas a beam of the `beam_width` highest scoring
AttackedTexts, greedily updating the beam with the highest scoring
transformations from the current beam.
"""

from typing import NoReturn, List

import numpy as np

from ..goal_functions import GoalFunctionResult, GoalFunctionResultStatus
from .base import SearchMethod


__all__ = [
    "BeamSearch",
    "GreedySearch",
]


class BeamSearch(SearchMethod):
    """
    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation (Transformation): The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    __name__ = "BeamSearch"

    def __init__(self, beam_width: int = 8, verbose: int = 0) -> NoReturn:
        """ """
        self.beam_width = beam_width
        self.verbose = verbose

    def perform_search(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        """ """
        if self.verbose >= 2:
            print(
                f"searching using {self.__name__} from initial result of example\n{initial_result.attacked_text.text}\n"
            )
        beam = [initial_result.attacked_text]
        best_result = initial_result
        n_round = 0
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            n_round += 1
            if self.verbose >= 2:
                print(f"in the {n_round}-th searching round")
                print(f"start size (width) of beam = {len(beam)}")
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result

    @property
    def is_black_box(self) -> bool:
        return True

    def extra_repr_keys(self) -> List[str]:
        return ["beam_width"]


class GreedySearch(BeamSearch):
    """A search method that greedily chooses from a list of possible perturbations.

    Implemented by calling ``BeamSearch`` with beam_width set to 1.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(beam_width=1, verbose=verbose)

    def extra_repr_keys(self):
        return []
