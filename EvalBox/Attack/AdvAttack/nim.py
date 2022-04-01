import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd

from EvalBox.Attack.AdvAttack.attack import Attack
from Models.resnet_cifar10 import resnet18

import cv2
from torch.nn.functional import interpolate as scale_image

import os


class NIM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        
        super(NIM, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.surrogate_model = resnet18()
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
            decay_factor:
        } 
        @return: None
        """

        self.eps = float(kwargs.get("epsilon", 0.1))
        self.num_steps = int(kwargs.get("num_steps", 20))
        self.eps_iter = float(kwargs.get("eps_iter", 0.01))
        model_s = str(kwargs.get("model_name", "torchvision.models.resnet18"))
        model_path = str(kwargs.get("model_path", ""))
        self.surrogate_model = self.get_model(model_path, model_s, self.device)
        self.decay_factor = float(kwargs.get("decay_factor", 0.9))
    
    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        momentum = np.zeros_like(copy_xs)
        targeted = self.IsTargeted

        for _ in range(self.num_steps):
            var_xs = Variable(
                torch.from_numpy(copy_xs+self.eps_iter*self.decay_factor*momentum).float().to(device), requires_grad=True
            )
            var_ys = Variable(ys.to(device))

            outputs = self.surrogate_model(var_xs)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            else:

                loss = self.criterion(outputs, var_ys)
            loss.backward()

            grad = var_xs.grad.data.cpu().numpy()
            mean = np.mean(np.abs(grad), (1,2,3), keepdims=True)
            grad = grad/mean

            momentum = self.decay_factor * momentum + grad

            copy_xs = copy_xs + self.eps_iter * np.sign(momentum)
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs
    
    
