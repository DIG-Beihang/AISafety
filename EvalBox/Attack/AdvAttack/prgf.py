import numpy as np
from numpy.lib.function_base import diff
from scipy.stats import stats
import torch
from torch.autograd import Variable
import torch.autograd as autograd

from EvalBox.Attack.AdvAttack.attack import Attack
from Models.resnet_cifar10 import resnet18

import cv2
from torch.nn.functional import interpolate as scale_image
import scipy.stats
import os


class PRGF(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        
        super(PRGF, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()
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
        self.lr = float(kwargs.get("lr", 0.01))
        self.max_queries = int(kwargs.get("max_queries", 200))
        model_s = str(kwargs.get("model_name", "torchvision.models.resnet18"))
        model_path = str(kwargs.get("model_path", ""))
        self.surrogate_model = self.get_model(model_path, model_s, self.device)
        self.norm = str(kwargs.get("norm", "linf"))
        self.method = str(kwargs.get("method", "biased"))
        self.fixed_const = float(kwargs.get("fixed_const", 0.5))
        self.dataprior = int(kwargs.get("dataprior", 0))
        self.samples_per_draw = int(kwargs.get("samples_per_draw", 50))

    def get_logits(self, xs):
        var_xs = torch.tensor(
            xs, dtype=torch.float, device=self.device, requires_grad=False
        )
        if len(var_xs.shape)==3:
            var_xs = var_xs.unsqueeze(0)
        return self.model(var_xs).detach().to("cpu")
    def get_loss(self, xs, ys):
        var_ys = torch.tensor(ys)
        if len(var_ys.shape)==0:
            var_ys = var_ys.unsqueeze(0)
        return self.criterion(self.get_logits(xs), var_ys).numpy()
    def get_pred(self, xs):
        return np.argmax(self.get_logits(xs).numpy(), 1)

    def get_s_grad(self, xs, ys):
        if len(xs.shape)==3:
            xsc = np.expand_dims(xs, 0)
        else:
            xsc = xs
        var_xs = torch.tensor(
            xsc, dtype=torch.float, device=self.device, requires_grad=True
        )
        var_ys = torch.tensor(ys)
        if len(var_ys.shape)==0:
            var_ys = var_ys.unsqueeze(0)
        loss = self.criterion(self.surrogate_model(var_xs), var_ys.to(self.device))
        loss.backward()
        grad = var_xs.grad.data.cpu().numpy()
        return grad
    
    def get_surrogate_logits(self, xs):
        var_xs = torch.tensor(
            xs, dtype=torch.float, device=self.device, requires_grad=False
        )
        return self.surrogate_model(var_xs).to(xs.device)

    def attack_single(self, x, y):

        image_size = x.shape[1]
        
        # ---Setting hyperparameters---
        if self.norm == 'l2':
            epsilon = 1e-3
            eps = np.sqrt(epsilon * image_size * image_size * 3)
            learning_rate = 2.0 / np.sqrt(image_size * image_size * 3)
        else:
            epsilon = self.eps
            eps = epsilon
            learning_rate = self.lr
        
        ini_sigma = 1e-4
        # -----------------------------
        
        sigma = ini_sigma
        # np.random.seed(0)
        # tf.set_random_seed(0)
        image = x
        adv_image = image.copy()
        label = self.get_pred(image)
        l = self.get_loss(image, label)
        if label[0]!=y:
            return x

        lr = learning_rate
        last_loss = []
        total_q = 0
        ite = 0

        while total_q <= self.max_queries:
            total_q += 1

            if ite % 2 == 0 and sigma != ini_sigma:
                # print("sigma has been increased before; checking if sigma could be set back to ini_sigma")
                rand = np.random.normal(size=adv_image.shape)
                rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
                rand_loss = self.get_loss(adv_image + ini_sigma * rand, label)
                total_q += 1
                rand = np.random.normal(size=adv_image.shape)
                rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
                rand_loss2 = self.get_loss(adv_image + ini_sigma * rand, label)
                total_q += 1
                if (rand_loss - l) != 0 and (rand_loss2 - l) != 0:
                    # print("set sigma back to ini_sigma")
                    sigma = ini_sigma

            if self.method != 'uniform':
                s_label = label
                prior = np.squeeze(self.get_s_grad(adv_image, s_label))
                prior = prior / np.maximum(1e-12, np.sqrt(np.mean(np.square(prior))))
            
            if self.method in ['biased', 'average']:
                start_iter = 3
                if ite % 10 == 0 or ite == start_iter:
                    # Estimate norm of true gradient periodically when ite == 0/10/20...;
                    # since gradient norm may change fast in the early iterations, we also
                    # estimate the gradient norm when ite == 3.
                    s = 10
                    pert = np.random.normal(size=(s,) + adv_image.shape)
                    for i in range(s):
                        pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
                    eval_points = adv_image + sigma * pert
                    losses = self.get_loss(eval_points, np.repeat(label, s))
                    total_q += s
                    norm_square = np.average(((losses - l) / sigma) ** 2)

                while True:
                    prior_loss = self.get_loss(adv_image + sigma * prior, label)
                    # print(prior_loss, sigma, l, label, prior)
                    total_q += 1
                    diff_prior = (prior_loss - l)
                    if diff_prior == 0:
                        # Avoid the numerical issue in finite difference
                        sigma *= 2
                        print("multiply sigma by 2")
                    else:
                        break

                est_alpha = diff_prior / sigma / np.maximum(np.sqrt(np.sum(np.square(prior)) * norm_square), 1e-12)
                # print("Estimated alpha =", est_alpha)
                alpha = est_alpha
                if alpha < 0:
                    prior = -prior
                    alpha = -alpha

            q = self.samples_per_draw
            n = image_size * image_size * 3
            d = 50*50*3
            gamma = 3.5
            A_square = d / n * gamma

            return_prior = False
          
            if self.method == 'biased':
                if self.dataprior:
                    best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                                    A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
                else:
                    best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                                    alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
                # print('best_lambda = ', best_lambda)
                if best_lambda < 1 and best_lambda > 0:
                    lmda = best_lambda
                else:
                    if alpha ** 2 * (n + 2 * q - 2) < 1:
                        lmda = 0
                    else:
                        lmda = 1
                if np.abs(alpha) >= 1:
                    lmda = 1
                # print('lambda = ', lmda)
                if lmda == 1:
                    return_prior = True
            elif self.method == 'fixed_biased':
                lmda = self.fixed_const

            if not return_prior:
                if self.dataprior:
                    pert = np.random.normal(size=(q, 50, 50, 3))
                    pert = np.array([cv2.resize(pert[i], adv_image.shape[:2],
                                                                            interpolation=cv2.INTER_NEAREST) for i in range(q)])
                else:
                    pert = np.random.normal(size=(q,) + adv_image.shape)
                for i in range(q):
                    if self.method in ['biased', 'fixed_biased']:
                        pert[i] = pert[i] - np.sum(pert[i] * prior) * prior / np.maximum(1e-12,
                            np.sum(prior * prior))
                        pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
                        pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior
                    else:
                        pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))

                while True:
                    eval_points = adv_image + sigma * pert
                    losses = self.get_loss(eval_points, np.repeat(label, q))
                    total_q += q

                    grad = (losses - l).reshape(-1,1,1,1) * pert
                    grad = np.mean(grad, axis=0)
                    norm_grad = np.sqrt(np.mean(np.square(grad)))
                    if norm_grad == 0:
                        # Avoid the numerical issue in finite difference
                        sigma *= 5
                        # print("estimated grad == 0, multiply sigma by 5")
                    else:
                        break
                grad = grad / np.maximum(1e-12, np.sqrt(np.mean(np.square(grad))))     
                final = grad      
            else:
                final = prior

            if self.norm == 'l2':
                adv_image = adv_image + lr * final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
                norm = max(1e-12, np.linalg.norm(adv_image - image))
                factor = min(1, eps / norm)
                adv_image = image + (adv_image - image) * factor
            else:
                adv_image = adv_image + lr * np.sign(final)
                adv_image = np.clip(adv_image, image - eps, image + eps)
            adv_image = np.clip(adv_image, 0, 1)

            adv_label = self.get_pred(adv_image)
            l = self.get_loss(adv_image, label)

            # print('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_label,
            #     'distortion:', np.max(np.abs(adv_image - image)), np.linalg.norm(adv_image - image))

            ite += 1

            if adv_label[0] != label[0]:
                print('Stop at queries:', total_q)
                return adv_image
        print(f"attack fail in {self.max_queries} queries")
        return adv_image

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        xs_adv = torch.clone(xs)
        batch_size = xs.shape[0]
        for i in range(batch_size):
            x_adv = self.attack_single(xs[i].numpy(), ys[i])
            xs_adv[i] = torch.tensor(x_adv)
        return xs_adv
