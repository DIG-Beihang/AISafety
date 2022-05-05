import re
from abc import ABC, abstractmethod
from typing import Sequence, NoReturn, Any, List

from utils.strings import ReprMixin

from abc import ABC, abstractmethod
class AudioModel(ABC):
    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()
    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError()


class NLPVictimModel(ReprMixin, ABC):
    """

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    __name__ = "NLPVictimModel"

    @abstractmethod
    def __call__(self, text_input_list: Sequence[str], **kwargs: Any) -> NoReturn:
        raise NotImplementedError()

    def get_grad(self, text_input: str) -> Any:
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method for `tokenize`"""
        raise NotImplementedError()

    def tokenize(
        self, inputs: Sequence[str], strip_prefix: bool = False
    ) -> List[List[str]]:
        """Helper method that tokenizes input strings
        Args:
            inputs: list of input strings
            strip_prefix: If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens: List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            strip_pattern = f"^({'|'.join(strip_chars)})"
            tokens = [[re.sub(strip_pattern, "", t) for t in x] for x in tokens]
        return tokens
