"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-11-10
@LastEditTime: 2022-03-05
"""

from abc import ABC, abstractmethod
from typing import Any, List

from ...utils.strings import ReprMixin


__all__ = [
    "AdvDetector",
]


class AdvDetector(ReprMixin, ABC):
    """ """

    __name__ = "AdvDetector"

    @abstractmethod
    def detect(self, text: str) -> Any:
        """ """
        raise NotImplementedError

    def __call__(self, text: str) -> Any:
        """
        alias of `self.detect`
        """
        return self.detect(text)

    def extra_repr_keys() -> List[str]:
        """ """
        return []
