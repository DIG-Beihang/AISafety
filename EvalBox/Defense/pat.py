#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-12 15:56:16
@LastEditTime: 2019-04-15 17:33:33
"""
import numpy as np
import os
import torch
from torch.autograd import Variable
from utils.Defense_utils import adjust_learning_rate
from EvalBox.Defense.defense import Defense


class PAT(Defense):
    def __init__(
        self, model=None, device=None, optimizer=None, scheduler=None, **kwargs
    ):
        """
        @description: PGD-based adversarial training (PAT)
        @param {
            model:
            device:
            optimizer:
            scheduler:
            kwargs:
        } 
        @return: None
        """
        super().__init__(model, device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            num_epochs:
            epsilon:
        } 
        @return: None
        """
        self.attack_step_num = int(kwargs.get("attack_step_num", 40))
        self.num_epochs = int(kwargs.get("num_epochs", 100))
        self.step_size = float(kwargs.get("step_size", 0.01))
        self.epsilon = float(kwargs.get("epsilon", 0.3))

    def _pgd_generation(self, var_natural_images=None, var_natural_labels=None):
        """
        @description: 
        @param {
            var_natural_images:
            var_natural_labels:
        } 
        @return: adv_images
        """
        self.model.eval()
        natural_images = var_natural_images.cpu().numpy()

        copy_images = natural_images.copy()
        copy_images = copy_images + np.random.uniform(
            -self.epsilon, self.epsilon, copy_images.shape
        ).astype("float32")

        for i in range(self.attack_step_num):
            var_copy_images = torch.from_numpy(copy_images).to(self.device)
            var_copy_images.requires_grad = True

            preds = self.model(var_copy_images)
            loss = self.criterion(preds, var_natural_labels)
            gradient = torch.autograd.grad(loss, var_copy_images)[0]
            gradient_sign = torch.sign(gradient).cpu().numpy()

            copy_images = copy_images + self.step_size * gradient_sign

            copy_images = np.clip(
                copy_images,
                natural_images - self.epsilon,
                natural_images + self.epsilon,
            )
            copy_images = np.clip(copy_images, 0.0, 1.0)

        return torch.from_numpy(copy_images).to(self.device)

    def valid(self, valid_loader=None):
        """
        @description: 
        @param {
            valid_loader:
            epoch:
        } 
        @return: val_acc
        """
        device = self.device
        self.model.to(device).eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, 1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
            val_acc = correct / total
        return val_acc

    def train(self, train_loader=None, epoch=None):
        """
        @description: 
        @param {
            train_loader:
            epoch:
        } 
        @return: None
        """
        device = self.device
        self.model.to(device)

        for index, (images, labels) in enumerate(train_loader):
            pat_images = images.to(device)
            pat_labels = labels.to(device)

            self.model.eval()
            adv_images = self._pgd_generation(
                var_natural_images=pat_images, var_natural_labels=pat_labels
            )

            self.model.train()

            logits_pat = self.model(pat_images)
            loss_pat = self.criterion(logits_pat, pat_labels)

            logits_adv = self.model(adv_images)
            loss_adv = self.criterion(logits_adv, pat_labels)

            loss = 0.5 * (loss_pat + loss_adv)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                "\rTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  \tloss_pat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> ".format(
                    epoch,
                    index,
                    len(train_loader),
                    loss_pat.item(),
                    loss_adv.item(),
                    loss.item(),
                ),
                end=" ",
            )

    def generate(
        self, train_loader=None, valid_loader=None, defense_enhanced_saver=None
    ):
        """
        @description: 
        @param {
            train_loader:
            valid_loader:
        } 
        @return: best_model_weights, best_acc
        """
        best_val_acc = None
        best_model_weights = self.model.state_dict()
        dir_path = os.path.dirname(defense_enhanced_saver)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for epoch in range(self.num_epochs):

            self.train(train_loader, epoch)
            val_acc = self.valid(valid_loader)

            adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)

            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()
                self.model.save(name=defense_enhanced_saver)
            else:
                print(
                    "Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n".format(
                        epoch, best_val_acc
                    )
                )

        print("Best val Acc: {:.4f}".format(best_val_acc))
        return best_model_weights, best_val_acc
