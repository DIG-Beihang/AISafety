import torch
import torch.nn as nn
from .attack import Attacker
from .utils import target_sentence_to_label
from tqdm import tqdm

class PGDAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(PGDAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()
    def _parse_params(self, **kwargs):
        self.eps = kwargs.get('eps', 0.025)
        self.iteration = kwargs.get('iteration', 50)
        self.alpha = kwargs.get('alpha', 1e-3)
    def generate(self, sounds, targets):
        targets = target_sentence_to_label(targets)
        targets = targets.view(1,-1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        advs = sounds.clone().detach().to(self.device).requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            for i in tqdm(range(self.iteration)):
                self.model.zero_grad()
                out, output_sizes = self.model(advs)
                out = out.transpose(0, 1).log()
                loss = self.criterion(out, targets, output_sizes, target_lengths)
                loss.backward()
                data_grad = advs.grad.data.nan_to_num(nan=0)
                advs = advs - self.alpha * data_grad.sign()
                noise = torch.clamp(advs - sounds.data, min=-self.eps, max=self.eps)
                advs = sounds + noise
                advs = torch.clamp(advs, min=-1, max=1)
                advs = advs.detach().requires_grad_(True)
        return advs