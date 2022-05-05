# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-27
@LastEditTime: 2022-03-19

主要参考了TextAttack的实现方式，同时参考了OpenAttack的部分实现
"""

from abc import ABC, abstractmethod
from typing import Sequence, NoReturn, List

from ..attacked_text import AttackedText
from ..transformations.base import Transformation
from ..strings import ReprMixin


__all__ = [
    "Constraint",
    "PreTransformationConstraint",
]


class Constraint(ReprMixin, ABC):
    """用于检查生成的文本对抗样本是否符合某些限制条件的抽象基类"""

    __name__ = "Constraint"

    def __init__(self, compare_against_original: bool) -> NoReturn:
        """
        @param {
            compare_against_original: 是否与原始文本进行比较；若否，则与上一个对抗样本进行比较
        }
        @return: None
        """
        self.compare_against_original = compare_against_original

    def check_batch(
        self, transformed_texts: Sequence[AttackedText], reference_text: AttackedText
    ) -> List[AttackedText]:
        """
        @description: 批量检查对抗样本是否符合限制条件
        @param {
            transformed_texts: 待检查的对抗样本序列
            reference_text: 基准文本
        }
        @return: 符合限制条件的对抗样本列表
        """
        incompatible_transformed_texts = []
        compatible_transformed_texts = []
        for transformed_text in transformed_texts:
            try:
                if self.check_compatibility(
                    transformed_text.attack_attrs["last_transformation"]
                ):
                    compatible_transformed_texts.append(transformed_text)
                else:
                    incompatible_transformed_texts.append(transformed_text)
            except KeyError:
                raise KeyError(
                    "transformed_text must have `last_transformation` attack_attr to apply constraint"
                )
        filtered_texts = self._check_batch(compatible_transformed_texts, reference_text)
        return list(filtered_texts) + incompatible_transformed_texts

    def _check_batch(
        self, transformed_texts: Sequence[AttackedText], reference_text: AttackedText
    ) -> List[AttackedText]:
        """ """
        return [
            transformed_text
            for transformed_text in transformed_texts
            if self._check_constraint(transformed_text, reference_text)
        ]

    def check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """
        @description: 检查对抗样本是否符合限制条件
        @param {
            transformed_texts: 待检查的对抗样本
            reference_text: 基准文本
        }
        @return: 符合限制条件的对抗样本
        """
        if not isinstance(transformed_text, AttackedText):
            raise TypeError("transformed_text must be of type AttackedText")
        if not isinstance(reference_text, AttackedText):
            raise TypeError("reference_text must be of type AttackedText")

        try:
            if not self.check_compatibility(
                transformed_text.attack_attrs["last_transformation"]
            ):
                return True
        except KeyError:
            raise KeyError(
                "`transformed_text` must have `last_transformation` attack_attr to apply constraint."
            )
        return self._check_constraint(transformed_text, reference_text)

    def __call__(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        return self.check_constraint(transformed_text, reference_text)

    @abstractmethod
    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """ """
        raise NotImplementedError

    def check_compatibility(self, transformation: Transformation) -> bool:
        """Checks if this constraint is compatible with the given
        transformation. For example, the ``WordEmbeddingDistance`` constraint
        compares the embedding of the word inserted with that of the word
        deleted. Therefore it can only be applied in the case of word swaps,
        and not for transformations which involve only one of insertion or
        deletion.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return True

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        return ["compare_against_original"]


class PreTransformationConstraint(ReprMixin, ABC):
    """用于生成对抗样本前检查生成限制条件的抽象基类"""

    __name__ = "PreTransformationConstraint"

    def __call__(
        self, current_text: AttackedText, transformation: Transformation
    ) -> List[int]:
        """
        @param {
            current_text: 待检查的文本（将从此文本生成对抗样本）
            transformation: 将要进行的变换
        }
        @return: `current_text` 中符合要求的词的下标的列表
        """
        if not self.check_compatibility(transformation):
            return set(range(len(current_text.words)))
        return self._get_modifiable_indices(current_text)

    @abstractmethod
    def _get_modifiable_indices(self, current_text: AttackedText) -> List[int]:
        """Returns the word indices in ``current_text`` which are able to be
        modified. Must be overridden by specific pre-transformation
        constraints.

        Args:
            current_text: The ``AttackedText`` input to consider.
        """
        raise NotImplementedError

    def check_compatibility(self, transformation: Transformation) -> bool:
        """Checks if this constraint is compatible with the given
        transformation. For example, the ``WordEmbeddingDistance`` constraint
        compares the embedding of the word inserted with that of the word
        deleted. Therefore it can only be applied in the case of word swaps,
        and not for transformations which involve only one of insertion or
        deletion.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return True

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        return []
