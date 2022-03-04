from math import inf
from numpy.core.defchararray import decode
import torch
from torch import optim
from torch._C import DeviceObjType
import torch.nn as nn
from .attack import Attacker
from .utils import target_sentence_to_label
from tqdm import tqdm
import numpy as np
import scipy

class ImperceptibleCWAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(ImperceptibleCWAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()
    def _parse_params(self, **kwargs):
        self.eps = kwargs.get('eps', 0.1)
        self.max_iter_1 = kwargs.get('max_iter_1', 200)
        self.max_iter_2 = kwargs.get('max_iter_2', 2000)
        self.lr_1 = kwargs.get('lr_1', 1e-3)
        self.lr_2 = kwargs.get('lr_2', 5e-5)
        self.decrease_factor_eps = kwargs.get('decrease_factor_eps', 0.8)
        self.num_iter_decrease_eps = kwargs.get('num_iter_decrease_eps', 20)
        self.alpha = kwargs.get('alpha', 0.01)
        self.decrease_factor_alpha = kwargs.get('decrease_factor_alpha', 0.8)
        self.num_iter_decrease_alpha = kwargs.get('num_iter_decrease_alpha', 50)
        self.increase_factor_alpha = kwargs.get('increase_factor_alpha', 1.2)
        self.num_iter_increase_alpha = kwargs.get('num_iter_increase_alpha', 20)
        self.freq_disp = kwargs.get('freq_disp', 10)
        self.win_length = kwargs.get('win_length', 2048)
        self.hop_length = kwargs.get('hop_length', 512)
        self.n_fft = kwargs.get('n_fft', 2048)
        self.sample_rate = kwargs.get('sample_rate', 16384)
    def loss_th(self, delta, theta, original_max_psd):
        relu = torch.nn.ReLU()

        psd_transform_delta = self._psd_transform(
            delta, original_max_psd
        )

        return torch.mean(relu(psd_transform_delta - torch.tensor(theta).to(self.device)))
    def _attack_1st_stage(self, sounds, targets, raw_targets):
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        advs = sounds.clone().detach().requires_grad_(True)
        eps = torch.ones((sounds.shape[0], 1)).to(self.device) * self.eps
        minx = torch.clamp(sounds - eps, min=-1)
        maxx = torch.clamp(sounds + eps, max=1)
        optimizer = torch.optim.Adam([advs], lr=self.lr_1)
        results = advs.clone().detach()
        with torch.backends.cudnn.flags(enabled=False):
            for i in tqdm(range(self.max_iter_1)):
                optimizer.zero_grad()
                decode_out, out, output_sizes = self.model(advs, decode=True)
                decode_out = [x[0] for x in decode_out]
                out = out.transpose(0, 1).log()
                loss = self.criterion(out, targets, output_sizes, target_lengths)
                loss.backward()
                advs.grad.nan_to_num_(nan=0)
                optimizer.step()
                advs.data.clamp_(min=minx, max=maxx)
                if i % self.freq_disp == 0:
                    print(loss.item(), (results - sounds).abs().max(), (advs - sounds).abs().max(),
                        decode_out, raw_targets, [x == raw_targets for x in decode_out])
                if i % self.num_iter_decrease_eps == 0:
                    for j in range(len(decode_out)):
                        if decode_out[j] == raw_targets:
                            norm = (advs[j] - sounds[j]).abs().max()
                            if eps[j, 0] > norm:
                                eps[j, 0] = norm
                                results[j] = advs[j].clone()
                            eps[j, 0] = norm * self.decrease_factor_eps
                minx = torch.clamp(sounds - eps, min=-1)
                maxx = torch.clamp(sounds + eps, max=1)
        return results
    def _attack_2nd_stage(self, sounds, advs, targets, raw_targets, theta, original_max_psd):
        advs = advs.detach().requires_grad_(True)
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        minloss = [np.inf for _ in range(advs.shape[0])]
        alpha = torch.ones((sounds.shape[0], 1)).to(self.device) * self.alpha
        optimizer = torch.optim.Adam([advs], lr=self.lr_2)
        results = advs.clone().detach()
        with torch.backends.cudnn.flags(enabled=False):
            for i in tqdm(range(self.max_iter_2)):
                optimizer.zero_grad()
                decode_out, out, output_sizes = self.model(advs, decode=True)
                decode_out = [x[0] for x in decode_out]
                out = out.transpose(0, 1).log()
                loss_CTC = self.criterion(out, targets, output_sizes, target_lengths)
                loss_th = self.loss_th(advs - sounds.data, theta, original_max_psd)
                loss = loss_CTC + loss_th * alpha
                loss.backward()
                advs.grad.nan_to_num_(nan=0)
                optimizer.step()
                advs.data.clamp_(min=-1, max=1)
                if i % self.freq_disp == 0:
                    print(loss.item(), (results - sounds).abs().max(), (advs - sounds).abs().max(),
                        decode_out, raw_targets, decode_out[0] == raw_targets)
                for j in range(len(decode_out)):
                    if decode_out[j] == raw_targets:
                        if minloss[j] > loss_th.item():
                            minloss[j] = loss_th.item()
                            results[j] = advs[j].clone()
                        if i % self.num_iter_increase_alpha == 0:
                            alpha[j] *= self.increase_factor_alpha
                    else:
                        if i % self.num_iter_decrease_alpha == 0:
                            alpha[j] *= self.decrease_factor_alpha
        return results

    def generate(self, sounds, targets):
        assert sounds.shape[0] == 1
        theta, original_max_psd = self._compute_masking_threshold(sounds[0].data.cpu().numpy())
        theta = theta.transpose(1, 0)
        raw_targets = targets
        targets = target_sentence_to_label(targets)
        targets = targets.view(1,-1).to(self.device).detach()
        advs = sounds.clone().requires_grad_(True)
        advs = self._attack_1st_stage(sounds, targets, raw_targets)
        advs = self._attack_2nd_stage(sounds, advs, targets, raw_targets, theta, original_max_psd)
        return advs
    def _compute_masking_threshold(self, x):
        """
        Compute the masking threshold and the maximum psd of the original audio.

        :param x: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """
        import librosa

        # First compute the psd matrix
        # Get window for the transformation
        window = scipy.signal.get_window("hann", self.win_length, fftbins=True)

        # Do transformation
        transformed_x = librosa.core.stft(
            y=x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=False
        )
        transformed_x *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_x / self.win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * np.log10(psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        freqs = librosa.core.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float64) - np.inf
        bark_idx = np.argmax(barks > 1)
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []

            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float64)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s_array = np.array(t_s)

            theta.append(np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0))

        theta = np.array(theta)

        return theta, original_max_psd

    def _psd_transform(self, delta, original_max_psd):
        """
        Compute the psd matrix of the perturbation.

        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        import torch  # lgtm [py/repeated-import]

        # Get window for the transformation
        window_fn = torch.hann_window  # type: ignore

        # Return STFT of delta
        delta_stft = torch.stft(
            delta,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window_fn(self.win_length).to(self.device),
        ).to(self.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))

        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / self.win_length
        psd = psd ** 2
        psd = (
            torch.pow(torch.tensor(10.0).type(torch.float64), torch.tensor(9.6).type(torch.float64)).to(
                self.device
            )
            / torch.reshape(torch.tensor(original_max_psd).to(self.device), [-1, 1, 1])
            * psd.type(torch.float64)
        )

        return psd
