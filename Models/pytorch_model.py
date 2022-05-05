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