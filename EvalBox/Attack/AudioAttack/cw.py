from math import inf
import torch
from torch import optim
from torch._C import DeviceObjType
import torch.nn as nn
from .attack import Attacker
from .utils import target_sentence_to_label
from tqdm import tqdm

class CWAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(CWAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()
    def _parse_params(self, **kwargs):
        self.eps = kwargs.get('eps', 0.1)
        self.lambd = kwargs.get('lambd', 0)
        self.iteration = kwargs.get('iteration', 1000)
        self.lr = kwargs.get('learning_rate', 1e-3)
        self.decrease_factor = kwargs.get('decrease_factor', 0.8)
        self.num_iter_decrease_eps = kwargs.get('num_iter_decrease_eps', 10)
    def generate(self, sounds, targets):
        raw_targets = targets
        targets = target_sentence_to_label(targets)
        targets = targets.view(1,-1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        advs = sounds.clone().requires_grad_(True)
        eps = torch.ones((sounds.shape[0], 1)).to(self.device) * self.eps
        minx = torch.clamp(sounds - eps, min=-1)
        maxx = torch.clamp(sounds + eps, max=1)
        optimizer = torch.optim.Adam([advs], lr=self.lr)
        results = advs.clone().detach()
        with torch.backends.cudnn.flags(enabled=False):
            for i in tqdm(range(self.iteration)):
                optimizer.zero_grad()
                decode_out, out, output_sizes = self.model(advs, decode=True)
                decode_out = [x[0] for x in decode_out]
                out = out.transpose(0, 1).log()
                loss_CTC = self.criterion(out, targets, output_sizes, target_lengths)
                loss_norm = self.lambd * torch.mean((advs - sounds) ** 2)
                loss = loss_CTC + loss_norm
                loss.backward()
                advs.grad.nan_to_num_(nan=0)
                optimizer.step()
                advs.data.clamp_(min=minx, max=maxx)
                if i % self.num_iter_decrease_eps == 0:
                    print(loss.item(), (results - sounds).abs().max(), (advs - sounds).abs().max(),
                        decode_out, raw_targets, decode_out[0] == raw_targets)
                    for j in range(len(decode_out)):
                        if decode_out[j] == raw_targets:
                            norm = (advs[j] - sounds[j]).abs().max()
                            if eps[j, 0] > norm:
                                eps[j, 0] = norm
                                results[j] = advs[j].clone()
                            eps[j, 0] *= self.decrease_factor
                minx = torch.clamp(sounds - eps, min=-1)
                maxx = torch.clamp(sounds + eps, max=1)
                            
        return results