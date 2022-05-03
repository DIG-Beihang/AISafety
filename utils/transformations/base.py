# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-17
@LastEditTime: 2022-03-19

文本替换基类，主要基于TextAttack实现
"""

from abc import ABC, abstractmethod
import random
from string import ascii_letters
from typing import Sequence, List, NoReturn, Optional, Any

from ..attacked_text import AttackedText
from ..strings import ReprMixin


__all__ = [
    "Transformation",
    "CharSubstitute",
    "WordSubstitute",
    "WordInsertion",
    "WordDeletion",
    "WordMerge",
    "PhraseInsertion",
    "CompositeTransformation",
    "RandomCompositeTransformation",
    "transformation_consists_of",
    "transformation_consists_of_word_substitutes",
    "transformation_consists_of_word_substitutes_and_deletions",
]


class Transformation(ReprMixin, ABC):
    """文本替换的抽象基类，用于实现各种生成文本对抗样本的方法"""

    __name__ = "Transformation"

    def __call__(
        self,
        current_text: AttackedText,
        pre_transformation_constraints: list = [],
        indices_to_modify: Optional[list] = None,
        max_num: Optional[int] = None,
        shifted_idxs: bool = False,
    ) -> List[AttackedText]:
        """
        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)
        transformed_texts = self._get_transformations(
            current_text, indices_to_modify, max_num
        )
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

    @abstractmethod
    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: int = None,
    ) -> List[AttackedText]:
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError

    @property
    def deterministic(self) -> bool:
        return True


class PhraseInsertion(Transformation, ABC):
    """ """

    __name__ = "PhraseInsertion"

    @abstractmethod
    def _get_candidates(
        self,
        current_text: AttackedText,
        index: int,
        num: Optional[int] = None,
    ) -> List[List[str]]:
        """ """
        raise NotImplementedError

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        transformed_texts = []
        num = 0
        _max_num = max_num or float("inf")
        for i in indices_to_modify:
            new_phrases = self._get_candidates(current_text, i)

            new_transformted_texts = []
            for p in new_phrases:
                new_transformted_texts.append(
                    current_text.insert_text_before_word_index(i, p)
                )
                num += 1
                if num >= _max_num:
                    break
            transformed_texts.extend(new_transformted_texts)
            if num >= _max_num:
                return transformed_texts

        return transformed_texts


class WordSubstitute(Transformation, ABC):
    """An abstract class that takes a sentence and transforms it by replacing some of its words."""

    __name__ = "WordSubstitute"

    @abstractmethod
    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """Returns a set of candidate replacements given an input word. Must be overriden
        by specific word substitute transformations.

        Args:
            word: The input word to find replacements for.
            num: max number of candidates.
        """
        raise NotImplementedError

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        words = current_text.words
        transformed_texts = []
        num = 0
        _max_num = max_num or float("inf")
        for i in indices_to_modify:
            word_to_replace = words[i]
            try:
                pos_tag = current_text.pos_of_word_index(i)
            except Exception:
                pos_tag = None
            replacement_words = self._get_candidates(word_to_replace, pos_tag)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
                num += 1
                if num >= _max_num:
                    break
            transformed_texts.extend(transformed_texts_idx)
            if num >= _max_num:
                return transformed_texts

        return transformed_texts


class WordInsertion(Transformation):
    """ """

    __name__ = "WordInsertion"

    @abstractmethod
    def _get_candidates(
        self,
        current_text: AttackedText,
        index: int,
        num: Optional[int] = None,
    ) -> List[str]:
        """Returns a set of new words we can insert at position `index` of `current_text`
        Args:
            current_text: Current text to modify.
            index: Position in which to insert a new word
        Returns:
            list[str]: List of new words to insert.
        """
        raise NotImplementedError

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        """ """
        transformed_texts = []
        num = 0
        _max_num = max_num or float("inf")
        for i in indices_to_modify:
            new_words = self._get_candidates(current_text, i)

            new_transformted_texts = []
            for w in new_words:
                new_transformted_texts.append(
                    current_text.insert_text_before_word_index(i, w)
                )
                num += 1
                if num >= _max_num:
                    break
            transformed_texts.extend(new_transformted_texts)
            if num >= _max_num:
                return transformed_texts

        return transformed_texts


class WordDeletion(Transformation):
    """An abstract class that takes a sentence and transforms it by deleting a
    single word.

    letters_to_insert (string): letters allowed for insertion into words
    """

    __name__ = "WordDeletion"

    def _get_transformations(
        self,
        current_text: AttackedText,
        indices_to_modify: Sequence[int],
        max_num: Optional[int] = None,
    ) -> List[AttackedText]:
        # words = current_text.words
        transformed_texts = []
        if len(current_text.words) > 1:
            for i in indices_to_modify:
                transformed_texts.append(current_text.delete_word_at_index(i))
        return transformed_texts


class WordMerge(Transformation):
    """An abstract class for word merges."""

    __name__ = "WordMerge"

    def __call__(
        self,
        current_text: AttackedText,
        pre_transformation_constraints: list = [],
        indices_to_modify: Optional[Sequence[int]] = None,
        shifted_idxs: bool = True,
    ) -> List[AttackedText]:
        """Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls ``_get_transformations``.

        Args:
            current_text:
                The ``AttackedText`` to transform.
            pre_transformation_constraints:
                The ``PreTransformationConstraint`` to apply before beginning the transformation.
            indices_to_modify:
                Which word indices should be modified as dictated by the ``SearchMethod``.
            shifted_idxs:
                Whether indices have been shifted from their original position in the text.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words) - 1))
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            allowed_indices = constraint(current_text, self)
            for i in indices_to_modify:
                if i not in allowed_indices and i + 1 not in allowed_indices:
                    indices_to_modify.remove(i)

        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
            if len(text.attack_attrs["newly_modified_indices"]) == 0:
                print("xcv", text, len(text.attack_attrs["newly_modified_indices"]))
        return transformed_texts

    def _get_candidates(self, current_text: AttackedText, index: int) -> List[str]:
        """Returns a set of new words we can insert at position `index` of `current_text`
        Args:
            current_text: Current text to modify.
            index: Position in which to insert a new word
        Returns:
            list[str]: List of new words to insert.
        """
        raise NotImplementedError

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: Sequence[int]
    ) -> List[AttackedText]:
        """
        Return a set of transformed texts obtained by insertion a new word in `indices_to_modify`

        Args:
            current_text: Current text to modify.
            indices_to_modify: List of positions in which to insert a new word.
        Returns:
            list[AttackedText]: List of transformed texts
        """
        transformed_texts = []

        for i in indices_to_modify:
            new_words = self._get_candidates(current_text, i)

            for w in new_words:
                temp_text = current_text.replace_word_at_index(i, w)
                transformed_texts.append(temp_text.delete_word_at_index(i + 1))

        return transformed_texts


class CharSubstitute(WordSubstitute, ABC):
    """ """

    __name__ = "CharSubstitute"

    def __init__(
        self,
        random_one: bool = True,
        skip_first_char: bool = False,
        skip_last_char: bool = False,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Args:
            random_one:
                Whether to return a single word with two characters swapped.
                If not, returns all possible options.
            skip_first_char:
                Whether to disregard perturbing the first character.
            skip_last_char:
                Whether to disregard perturbing the last character.
        """
        # super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
        self.verbose = kwargs.get("verbose", 0)

    @abstractmethod
    def _get_candidates(
        self,
        word: str,
        pos_tag: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[str]:
        """Returns a set of candidate replacements given an input char. Must be overriden
        by specific char substitute transformations.

        Args:
            char: The input char to find replacements for.
            num: max number of candidates.
        """
        raise NotImplementedError

    def _get_random_letter(
        self, letters_to_insert: Sequence[str] = ascii_letters
    ) -> str:
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(letters_to_insert)


class CompositeTransformation(Transformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.
    """

    __name__ = "CompositeTransformation"

    def __init__(self, transformations: Sequence[Transformation]) -> NoReturn:
        """
        Args:
        transformations: The list of ``Transformation`` to apply.
        """
        if not isinstance(transformations, Sequence):
            raise TypeError(
                "transformations must be sequence of `Transform`, e.g. list, tuple, etc."
            )
        elif not len(transformations):
            raise ValueError("transformations cannot be empty")
        self.transformations = transformations

    def _get_transformations(self, *_) -> List[AttackedText]:
        """Placeholder method that would throw an error if a user tried to
        treat the CompositeTransformation as a 'normal' transformation."""
        raise RuntimeError(
            "CompositeTransformation does not support _get_transformations()."
        )

    def __call__(self, *args, **kwargs) -> List[AttackedText]:
        new_attacked_texts = set()
        for transformation in self.transformations:
            # print(f"args: {args}, kwargs: {kwargs}")
            # print(f"transformation = {transformation}")
            new_attacked_texts.update(transformation(*args, **kwargs))
        return list(new_attacked_texts)

    def _random_nsteps(self, nsteps: int, *args, **kwargs) -> List[AttackedText]:
        """
        randomly apply n steps of transformations sampled from `self.transformations`
        """
        seq = random.choices(population=range(len(self.transformations)), k=nsteps)
        new_attacked_texts = set()
        for idx in seq:
            new_attacked_texts.update(self.transformations[idx](*args, **kwargs))
        return list(new_attacked_texts)

    def __repr__(self) -> str:
        main_str = f"{self.__name__}("
        transformation_lines = []
        for i, transformation in enumerate(self.transformations):
            transformation_lines.append(f"({i}): {transformation}")
        transformation_lines.append(")")
        main_str += "\n" + "\n".join(transformation_lines)
        return main_str

    __str__ = __repr__


class RandomCompositeTransformation(CompositeTransformation):
    """A transformation which randomly applies some of a list of transformations,
    returning a set of all optoins.
    """

    __name__ = "RandomCompositeTransformation"

    def __init__(
        self, transformations: Sequence[Transformation], nsteps: Optional[int] = None
    ) -> NoReturn:
        """
        Args:
        transformations: The list of ``Transformation`` to apply.
        """
        super().__init__(transformations)
        self.nsteps = nsteps or len(transformations)

    def __call__(self, *args, **kwargs) -> List[AttackedText]:
        return self._random_nsteps(self.nsteps, *args, **kwargs)


def transformation_consists_of(
    transformation: Transformation, transformation_classes: Sequence[type]
) -> bool:
    """判断实例``transformation``的仅来自``transformation_classes``中的类"""
    if isinstance(transformation, CompositeTransformation):
        for t in transformation.transformations:
            if not transformation_consists_of(t, transformation_classes):
                return False
        return True
    else:
        for transformation_class in transformation_classes:
            if isinstance(transformation, transformation_class):
                return True
        return False


def transformation_consists_of_word_substitutes(transformation: Transformation) -> bool:
    """判断实例``transformation``的仅包含词替换"""
    return transformation_consists_of(transformation, [WordSubstitute])


def transformation_consists_of_word_substitutes_and_deletions(
    transformation: Transformation,
) -> bool:
    """判断实例``transformation``的仅包含词替换与词删除"""
    return transformation_consists_of(transformation, [WordDeletion, WordSubstitute])
