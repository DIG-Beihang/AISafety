# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-10-28

TextBugger: Generating Adversarial Text Against Real-world Applications

https://arxiv.org/abs/1812.05271

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
    SentenceEncoderBase,
    MultilingualUSE,
    RepeatModification,
    StopwordModification,
)
from ...utils.goal_functions import UntargetedClassification
from ...utils.search_methods import WordImportanceRanking
from ...utils.transformations import (  # noqa: F401
    CompositeTransformation,
    WordEmbeddingSubstitute,
    CharacterHomoglyphSubstitute,
    NeighboringCharacterSubstitute,
    RandomCharacterDeletion,
    RandomCharacterInsertion,
)
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
    "TextBugger",
]


class TextBugger(Attack):
    """ """

    __name__ = "TextBugger"

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
            "use_threshold",
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

        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        _letters_to_insert = {
            LANGUAGE.ENGLISH: " ",
            LANGUAGE.CHINESE: "的",  # TODO: better choice?
        }
        max_candidates = kwargs.get("max_candidates", 5)
        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                RandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=_letters_to_insert[self._language],
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                RandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                NeighboringCharacterSubstitute(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                CharacterHomoglyphSubstitute(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordEmbeddingSubstitute(
                    embedding, max_candidates=max_candidates, verbose=verbose
                ),
            ]
        )

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
        use_threshold = kwargs.get("use_threshold", 0.8)
        start = time.time()
        print("loading universal sentence encoder...")
        use_constraint = MultilingualUSE(
            threshold=use_threshold,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        print(f"universal sentence encoder loaded in {time.time()-start:.2f} seconds")
        constraints.append(use_constraint)

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(self._model)

        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = WordImportanceRanking(wir_method="delete", verbose=verbose)

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
        if "use_threshold" in kwargs:
            for c in self.constraints:
                if isinstance(c, (SentenceEncoderBase, MultilingualUSE)):
                    c.threshold = kwargs.get("use_threshold")
        self.verbose = kwargs.get("verbose", self.verbose)

    def prepare_data(self):
        """ """
        raise NotImplementedError("请勿调用此方法")

    def generate(self):
        """ """
        raise NotImplementedError("请勿调用此方法")
