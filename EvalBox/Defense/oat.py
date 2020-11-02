#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-15 11:17:44
@LastEditTime: 2019-04-15 18:02:13
"""
import numpy as np
import os
import torch
from torch.autograd import Variable
from utils.Defense_utils import adjust_learning_rate
from EvalBox.Defense.defense import Defense


class OAT(Defense):
    def __init__(
        self, model=None, device=None, optimizer=None, scheduler=None, **kwargs
    ):
        """
        @description: Original adversarial training (OAT)
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
            alpha:
        } 
        @return: None
        """
        self.num_epochs = int(kwargs.get("num_epochs", 200))
        self.epsilon = float(kwargs.get("epsilon", 0.03))
        self.alpha = float(kwargs.get("alpha", 0.5))

    def _fgsm_generation(self, var_xs=None, var_ys=None):
        """
        @description: 
        @param {
            var_xs:
            var_ys:
        } 
        @return: adv_xs
        """
        device = self.device
        self.model.eval().to(device)

        adv_num = int(self.alpha * var_xs.shape[0])
        # print(adv_num, type(adv_num))

        X = torch.tensor(
            var_xs[:adv_num], dtype=torch.float, device=device, requires_grad=True
        )
        Y = torch.tensor(var_ys[:adv_num], device=device)

        outputs = self.model(X)
        loss = self.criterion(outputs, Y)
        loss.backward()

        X_adv = X.detach() + self.epsilon * torch.sign(X.grad)
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        adv_xs = torch.cat((X_adv, var_xs[adv_num:].clone().to(device)), 0)
        adv_ys = torch.cat((Y, var_ys[adv_num:].clone().to(device)), 0)

        return adv_xs, adv_ys

    def valid(self, train_loader=None, epoch=None):
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
            for inputs, labels in train_loader:
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
            nat_images = images.to(device)
            nat_labels = labels.to(device)

            self.model.eval()
            adv_images, adv_labels = self._fgsm_generation(
                var_xs=nat_images, var_ys=nat_labels
            )

            self.model.train()

            logits_adv = self.model(adv_images)
            loss = self.criterion(logits_adv, adv_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                "\rTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  total_loss={:.4f} ===> ".format(
                    epoch, index, len(train_loader), loss.item()
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
            #################
            torch.save(self.model.state_dict(), defense_enhanced_saver)
            print(
                "Train Epoch{:>3}: validation dataset accuracy {:.4f}\n".format(
                    epoch, val_acc
                )
            )
            #################
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()
                torch.save(self.model.state_dict(), defense_enhanced_saver)
            else:
                print(
                    "Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n".format(
                        epoch, best_val_acc
                    )
                )

        print("Best val Acc: {:.4f}".format(best_val_acc))
        return best_model_weights, best_val_acc
