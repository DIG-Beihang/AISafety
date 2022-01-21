from abc import ABC, ABCMeta, abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Attacker(object):
    __metaclass__ = ABCMeta
    def __init__(self, model, device):
        self.model = model
        self.device = device
    @abstractmethod
    def generate(self, sounds, targets):
        raise NotImplemented