# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-01
@LastEditTime: 2021-12-05

封装待评测的HuggingFace模型
"""

from typing import Sequence, NoReturn, Union, List

import torch
import transformers

from .pytorch_model import PyTorchNLPVictimModel


with torch.no_grad():
    torch.cuda.empty_cache()


__all__ = ["HuggingFaceNLPVictimModel"]


class HuggingFaceNLPVictimModel(PyTorchNLPVictimModel):
    """ """

    __name__ = "HuggingFaceNLPVictimModel"

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
    ) -> NoReturn:
        """ """
        self.model = model
        self.tokenizer = tokenizer
        self._pipeline = None

    def __call__(self, text_input_list: Sequence[str]) -> List[Union[str, float]]:
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as positional arguments.)
        """
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input: str) -> dict:
        """Get gradient of loss with respect to input text.

        Args:
            text_input: input string
        Returns:
            Dict of ids, and gradient as numpy array.
        """
        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiated your model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method that for `tokenize`
        Args:
            inputs: list of input strings
        Returns:
            tokens: List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

    @property
    def id2label(self) -> dict:
        """ """
        return self.model.config.id2label

    @property
    def max_length(self) -> int:
        """ """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        return (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
