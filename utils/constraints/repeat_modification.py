# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-22
@LastEditTime: 2021-08-22
"""

from typing import Set

from .base import PreTransformationConstraint
from ..attacked_text import AttackedText


class RepeatModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words which have already
    been modified."""

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        """Returns the word indices in current_text which are able to be
        deleted."""
        try:
            return (
                set(range(len(current_text.words)))
                - current_text.attack_attrs["modified_indices"]
            )
        except KeyError:
            raise KeyError(
                "`modified_indices` in attack_attrs required for RepeatModification constraint."
            )
