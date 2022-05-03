# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-21

Certified Robustness to Adversarial Word Substitutions

https://arxiv.org/abs/1909.00986

状态
-----
代码完成        √
手动测试完成    √
自动测试完成    X
文档完成        √
"""

import time
from typing import NoReturn, Any, Optional

import torch

from .attack import Attack
from ...Models.base import NLPVictimModel
from ...utils.constraints import (  # noqa: F401
    MaxWordsPerturbed,
    WordEmbeddingDistance,
    RepeatModification,
    StopwordModification,
    Learning2Write,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import AlzantotGeneticAlgorithm
from ...utils.transformations import WordEmbeddingSubstitute
from ...utils.assets import fetch
from ...utils.strings import normalize_language, LANGUAGE
from ...utils.word_embeddings import (  # noqa: F401
    GensimWordEmbedding,
    CounterFittedEmbedding,
    ChineseWord2Vec,
)
from ...Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from ...Models.TestModel.roberta_sst import VictimRoBERTaSST


__all__ = [
    "FasterGenetic",
]


class FasterGenetic(Attack):
    """ """

    __name__ = "FasterGenetic"

    def __init__(
        self,
        model: Optional[NLPVictimModel] = None,
        device: Optional[torch.device] = None,
        language: str = "zh",
        **kwargs: Any,
    ) -> NoReturn:
        """
        @description: The Boundary Attack
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        self.__initialized = False
        self.__valid_params = {
            "max_candidates",
            "max_perturb_percent",
            "max_mse_dist",
            "max_iters",
            "pop_size",
            "verbose",
        }
        self._language = normalize_language(language)
        self._model = model
        if not self._model:
            start = time.time()
            if self._language == LANGUAGE.CHINESE:
                print("load default Chinese victim model...")
                self._model = VictimBERTAmazonZH()
            else:
                print("load default English victim model...")
                self._model = VictimRoBERTaSST()
            print(f"default model loaded in {time.time()-start:.2f} seconds.")
        self._device = device
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """ """
        assert set(kwargs).issubset(self.__valid_params)
        if self.__initialized:
            self.__reset_params(**kwargs)
        self.__parse_params(**kwargs)
        self.__initialized = True

    def __parse_params(self, **kwargs):
        """
        @description:
        @param {
        }
        @return:
        """
        #
        # Section 5: Experiments
        #
        # We base our sets of allowed word substitutions S(x, i) on the
        # substitutions allowed by Alzantot et al. (2018). They demonstrated that
        # their substitutions lead to adversarial examples that are qualitatively
        # similar to the original input and retain the original label, as judged
        # by humans. Alzantot et al. (2018) define the neighbors N(w) of a word w
        # as the n = 8 nearest neighbors of w in a “counter-fitted” word vector
        # space where antonyms are far apart (Mrksiˇ c´ et al., 2016). The
        # neighbors must also lie within some Euclidean distance threshold. They
        # also use a language model constraint to avoid nonsensical perturbations:
        # they allow substituting xi with x˜i ∈ N(xi) if and only if it does not
        # decrease the log-likelihood of the text under a pre-trained language
        # model by more than some threshold.
        #
        # We make three modifications to this approach:
        #
        # First, in Alzantot et al. (2018), the adversary
        # applies substitutions one at a time, and the
        # neighborhoods and language model scores are computed.
        # Equation (4) must be applied before the model
        # can combine information from multiple words, but it can
        # be delayed until after processing each word independently.
        # Note that the model itself classifies using a different
        # set of pre-trained word vectors; the counter-fitted vectors
        # are only used to define the set of allowed substitution words.
        # relative to the current altered version of the input.
        # This results in a hard-to-define attack surface, as
        # changing one word can allow or disallow changes
        # to other words. It also requires recomputing
        # language model scores at each iteration of the genetic
        # attack, which is inefficient. Moreover, the same
        # word can be substituted multiple times, leading
        # to semantic drift. We define allowed substitutions
        # relative to the original sentence x, and disallow
        # repeated substitutions.
        #
        # Second, we use a faster language model that allows us to query
        # longer contexts; Alzantot et al. (2018) use a slower language
        # model and could only query it with short contexts.

        # Finally, we use the language model constraint only
        # at test time; the model is trained against all perturbations in N(w). This encourages the model to be
        # robust to a larger space of perturbations, instead of
        # specializing for the particular choice of language
        # model. See Appendix A.3 for further details. [This is a model-specific
        # adjustment, so does not affect the attack recipe.]
        #
        # Appendix A.3:
        #
        # In Alzantot et al. (2018), the adversary applies replacements one at a
        # time, and the neighborhoods and language model scores are computed
        # relative to the current altered version of the input. This results in a
        # hard-to-define attack surface, as the same word can be replaced many
        # times, leading to semantic drift. We instead pre-compute the allowed
        # substitutions S(x, i) at index i based on the original x. We define
        # S(x, i) as the set of x_i ∈ N(x_i) such that where probabilities are
        # assigned by a pre-trained language model, and the window radius W and
        # threshold δ are hyperparameters. We use W = 6 and δ = 5.
        #

        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        verbose = kwargs.get("verbose", 0)
        print("start initializing attacking parameters...")
        start = time.time()
        print("loading stopwords and word embedding...")
        if self._language == LANGUAGE.CHINESE:
            stopwords = fetch("stopwords_zh")
            embedding = ChineseWord2Vec()
        elif self._language == LANGUAGE.ENGLISH:
            stopwords = fetch("stopwords_en")
            embedding = CounterFittedEmbedding()
        else:
            raise ValueError(f"暂不支持语言 {self._language.name.lower()}")
        print(f"stopwords and word embedding loaded in {time.time()-start:.2f} seconds")

        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        #
        # Maximum words perturbed percentage of 20%
        #
        max_perturb_percent = kwargs.get("max_perturb_percent", 0.2)
        constraints.append(MaxWordsPerturbed(max_percent=max_perturb_percent))

        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        max_mse_dist = kwargs.get("max_mse_dist", 0.5)
        constraints.append(WordEmbeddingDistance(embedding, max_mse_dist=max_mse_dist))

        #
        # Language Model
        #
        constraints.append(
            Learning2Write(
                window_size=6, max_log_prob_diff=5.0, compare_against_original=True
            )
        )

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)

        #
        # Perform word substitution with a genetic algorithm.
        #
        pop_size = kwargs.get("pop_size", 60)
        max_iters = kwargs.get("max_iters", 20)
        search_method = AlzantotGeneticAlgorithm(
            pop_size=pop_size, max_iters=max_iters, post_crossover_check=False
        )

        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted Paragram Embeddings.
        #
        # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
        #
        max_candidates = kwargs.get("max_candidates", 8)
        transformation = WordEmbeddingSubstitute(
            embedding=embedding,
            max_candidates=max_candidates,
            verbose=verbose,
        )

        super().__init__(
            model=self._model,
            device=self._device,
            IsTargeted=False,
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method,
            language=self._language,
            verbose=verbose,
        )

    def __reset_params(self, **kwargs: Any) -> NoReturn:
        """ """
        if "max_candidates" in kwargs:
            self.transformation.max_candidates = kwargs.get("max_candidates")
        if "max_perturb_percent" in kwargs:
            for c in self.constraints:
                if isinstance(c, MaxWordsPerturbed):
                    c.max_percent = kwargs.get("max_perturb_percent")
        if "max_mse_dist" in kwargs:
            for c in self.constraints:
                if isinstance(c, WordEmbeddingDistance):
                    c.max_mse_dist = kwargs.get("max_mse_dist")
        if "pop_size" in kwargs:
            self.search_method.pop_size = kwargs.get("pop_size")
        if "max_iters" in kwargs:
            self.search_method.max_iters = kwargs.get("max_iters")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
