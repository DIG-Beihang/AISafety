import sys
sys.path.append('../')
from Models.base import AudioModel
from utils.stft import STFT, torch_spectrogram
import torch
class PyTorchAudioModel(AudioModel):
    def __init__(self, model, decoder, device, sample_rate=16000):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.decoder = decoder
        self.sample_rate = sample_rate
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
    def __call__(self, inputs, decode=False):
        spec = torch_spectrogram(inputs, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int().repeat(spec.size(0))
        out, output_sizes = self.model(spec, input_sizes)
        if decode:
            decode_out, decoded_offsets = self.decoder.decode(out, output_sizes)
            return decode_out, out, output_sizes
        else:
            return out, output_sizes
    def zero_grad(self):
        return self.model.zero_grad()

from typing import Callable, Sequence, NoReturn, Union, List

import numpy as np
import torch
import torch.nn as nn

from Models.base import NLPVictimModel


with torch.no_grad():
    torch.cuda.empty_cache()


class PyTorchNLPVictimModel(NLPVictimModel):
    """ """

    __name__ = "PyTorchNLPVictimModel"

    def __init__(
        self, model: nn.Module, tokenizer: Callable[[Sequence[str]], np.ndarray]
    ) -> NoReturn:
        """ """
        self.model = model
        self.tokenizer = tokenizer

    def to(self, device: torch.device) -> NoReturn:
        self.model.to(device)

    def __call__(
        self, text_input_list: Sequence[str], batch_size: int = 32
    ) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        outputs = []
        i = 0
        while i < len(ids):
            batch = ids[i : i + batch_size]
            with torch.no_grad():
                batch_preds = self.model(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]
            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu().detach().numpy()
            outputs.append(batch_preds)
            i += batch_size
        outputs = np.concatenate(outputs, axis=0)

        return outputs

    def get_grad(
        self,
        text_input: str,
        loss_fn: Union[nn.Module, Callable] = nn.CrossEntropyLoss(),
    ) -> dict:
        """Get gradient of loss with respect to input text.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

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
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method that for `tokenize`
        Args:
            inputs: list of input strings
        Returns:
            tokens: List of list of tokens as strings
        """
        tokens = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs
        ]
        return tokens
