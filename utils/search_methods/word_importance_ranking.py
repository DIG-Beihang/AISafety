# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-21
@LastEditTime: 2022-04-17

基于词在句子中重要性生成对抗样本的方法，
包括TextFooler, PWWS等都是基于这一类方法

"""

from typing import NoReturn, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax

from ..misc import DEFAULTS
from ..attacked_text import AttackedText
from ..transformations import transformation_consists_of_word_substitutes_and_deletions
from ..goal_functions import GoalFunctionResultStatus, GoalFunctionResult
from .base import SearchMethod


__all__ = [
    "WordImportanceRanking",
]


class WordImportanceRanking(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.
    """

    __name__ = "WordImportanceRanking"

    def __init__(self, wir_method: str = "unk", verbose: int = 0) -> NoReturn:
        """
        @param {
            wir_method: method for ranking most important words
        }
        @return: None
        """
        self.wir_method = wir_method.lower()
        self.verbose = verbose

    def _get_index_order(self, initial_text: AttackedText) -> Tuple[np.ndarray, bool]:
        """Returns word indices of ``initial_text`` in descending order of importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(initial_text.num_words)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, word in enumerate(initial_text.words):
                matched_tokens = word2token_mapping[i]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    try:
                        agg_grad = np.mean(gradient[matched_tokens], axis=0)
                        index_scores[i] = np.linalg.norm(agg_grad, ord=1)
                    except Exception:
                        index_scores[i] = 0.0

            search_over = False

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            DEFAULTS.RNG.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over

    def perform_search(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        """ """
        if self.verbose >= 2:
            print(
                f"searching using {self.__name__} from initial result of example\n{initial_result.attacked_text.text}\n"
            )
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            if self.verbose >= 2:
                print(f"in the {i+1}-th searching round")
                print(
                    f"number of transformed_text_candidates = {len(transformed_text_candidates)}"
                )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word substitues and deletion transformations."""
        return transformation_consists_of_word_substitutes_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
