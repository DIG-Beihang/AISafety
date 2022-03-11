from math import inf
from numpy.core.defchararray import decode
import torch
from torch import optim
import torch.nn.functional as F
from torch._C import DeviceObjType
import torch.nn as nn
from .attack import Attacker
from .utils import target_sentence_to_label, levenshteinDistance
from tqdm import tqdm
import numpy as np
from scipy.signal import butter, lfilter

def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)

class GeneticAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(GeneticAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
    def _parse_params(self, **kwargs):
        self.pop_size = kwargs.get('pop_size', 100)
        self.elite_size = kwargs.get('elite_size', 10)
        self.mutation_p = kwargs.get('mutation_p', 0.005)
        self.noise_stdev = kwargs.get('noise_stdev', 0.002) # \approx 40 / 16384
        self.momentum = kwargs.get('momentum', 0.9)
        self.alpha = kwargs.get('alpha', 0.001)
        self.iterations = kwargs.get('iterations', 3000)
        self.num_points_estimate = kwargs.get('num_points_estimate', 100)
        self.delta_for_gradient = kwargs.get('delta_for_gradient', 0.006) # \approx 100 / 16384
        self.delta_for_perturbation = kwargs.get('delta_for_perturbation', 0.06) # \approx 1000 / 16384
        self.freq_disp = kwargs.get('freq_disp', 10)
        self.decrease_factor = kwargs.get('decrease_factor', 0.995)
    def get_fitness_score(self, input_audio_batch, targets, target_lengths):
        input_audio_batch = torch.from_numpy(input_audio_batch).to(self.device).float()
        out, output_sizes = self.model(input_audio_batch)
        out = out.transpose(0, 1).log()
        targets = targets.repeat((input_audio_batch.size(0), 1))
        target_lengths = target_lengths.repeat((input_audio_batch.size(0), 1)).view(-1)
        scores = F.ctc_loss(out, targets, output_sizes, target_lengths, reduction='none')
        return -scores
    def get_text(self, input_audio):
        input_audio = torch.from_numpy(input_audio).to(self.device).float()
        return self.model(input_audio, decode=True)
    def get_new_pop(self, elite_pop, elite_pop_scores, pop_size):
        elite_logits = np.exp(elite_pop_scores - elite_pop_scores.max())
        elite_probs = elite_logits / elite_logits.sum()
        cand1 = elite_pop[np.random.choice(elite_pop.shape[0], p=elite_probs, size=pop_size)]
        cand2 = elite_pop[np.random.choice(elite_pop.shape[0], p=elite_probs, size=pop_size)]
        mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5
        next_pop = mask * cand1 + (1 - mask) * cand2
        return next_pop
    def mutate_pop(self, pop):
        noise = np.random.randn(*pop.shape) * self.noise_stdev
        noise = highpass_filter(noise)
        mask = np.random.randn(*pop.shape) < self.mutation_p
        new_pop = pop + noise * mask
        return new_pop
    def generate(self, sounds, targets):
        raw_targets = targets
        sounds = sounds.cpu().numpy()
        targets = target_sentence_to_label(targets)
        targets = targets.view(1,-1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1,-1)
        pop = np.tile(sounds, (self.pop_size, 1))
        prev_loss = None
        dist = np.inf
        mutation_p = self.mutation_p

        with torch.no_grad():
            for iter in tqdm(range(self.iterations)):
                pop_scores = self.get_fitness_score(pop, targets, target_lengths).cpu().numpy()
                elite_ind = np.argsort(pop_scores)[-self.elite_size:]
                elite_pop, elite_pop_scores = pop[elite_ind], pop_scores[elite_ind]
                if prev_loss is not None and prev_loss != elite_pop_scores[-1]:
                    mutation_p = self.momentum * mutation_p + self.alpha / np.abs(prev_loss - elite_pop_scores[-1])
                if iter % self.freq_disp == 0 or iter == self.iterations - 1:
                    print('Current loss: ', -elite_pop_scores[-1]) 
                    best_pop = elite_pop[None, -1]
                    best_text = self.get_text(best_pop)[0][0]
                    dist = levenshteinDistance(best_text, raw_targets)
                    print('{}; {}; {}'.format(best_text, raw_targets, dist))
                    if best_text == raw_targets:
                        break
                if dist > 2:
                    next_pop = self.get_new_pop(elite_pop, elite_pop_scores, self.pop_size)
                    pop = self.mutate_pop(next_pop)
                    prev_loss = elite_pop_scores[-1]
                else:
                    perturbed = np.tile(elite_pop[None, -1], (self.num_points_estimate, 1))
                    indices = np.random.choice(pop.shape[1], self.num_points_estimate, replace=False)
                    perturbed[np.arange(self.num_points_estimate), indices] += self.delta_for_gradient
                    perturbed_scores = self.get_fitness_score(perturbed, targets, target_lengths).cpu().numpy()
                    grad = (perturbed_scores - elite_pop_scores[-1]) / self.delta_for_gradient
                    grad = grad / np.abs(grad).max()
                    modified = elite_pop[-1].copy()
                    modified[indices] += grad * self.delta_for_perturbation
                    pop = np.tile(modified[None, :], (self.pop_size, 1))
                    self.delta_for_perturbation *= self.decrease_factor
        return torch.from_numpy(best_pop).to(self.device).float()
