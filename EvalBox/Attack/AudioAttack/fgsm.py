import torch
import torch.nn as nn
from .attack import Attacker
from .utils import target_sentence_to_label

class FGSMAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(FGSMAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()
    def _parse_params(self, **kwargs):
        self.eps = kwargs.get('eps', 0.025)
    def generate(self, sounds, targets):
        targets = target_sentence_to_label(targets)
        targets = targets.view(1,-1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        advs = sounds.clone().detach().to(self.device).requires_grad_(True)
        self.model.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):
            out, output_sizes = self.model(advs)
            out = out.transpose(0, 1).log()
            loss = self.criterion(out, targets, output_sizes, target_lengths)
            loss.backward()
            data_grad = advs.grad.data.nan_to_num(nan=0)
            advs = advs - self.eps * data_grad.sign()
            advs = advs.detach().requires_grad_(True)
        advs = torch.clamp(advs, min=-1, max=1)
        return advs