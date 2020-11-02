#!/usr/bin/env python
# coding=UTF-8
"""
@Author:
@LastEditors:
@Description:
@Date: 2019-06-21
@LastEditTime: 2019-06-21
"""
# !/usr/bin/env python
# coding=UTF-8

import numpy as np
import os
import torch
from torch.autograd import Variable
from utils.Defense_utils import adjust_learning_rate
from EvalBox.Defense.defense import Defense


class ANP(Defense):
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
        self.scheduler = scheduler

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
        self.num_epochs = int(kwargs.get("num_epochs", 120))
        self.enable_lat = bool(kwargs.get("enable_lat", True))
        self.epsilon = float(kwargs.get("epsilon", 0.6))
        self.alpha = float(kwargs.get("alpha", 0.6))
        self.batch_size = int(kwargs.get("batch_size", 64))

    def set_anp(self, epoch):
        if epoch < 40:
            return 0, 0, 1
        elif epoch < 60:
            return 0.3, 0.7, 3
        elif epoch < 80:
            return 1.0, 1.0, 3
        elif epoch < 100:
            return 0.3, 0.7, 3
        elif epoch < 120:
            return 0, 0, 1

    def valid(self, test_loader=None, epoch=None):
        """
        @description:
        @param {
            valid_loader:
            epoch:
        }
        @return: val_acc
        """
        if self.enable_lat:
            self.model.zero_reg()

        device = self.device
        self.model.to(device).eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, 1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
            val_acc = correct / total
        print("\nvalid after train {}/{} epochs".format(epoch + 1, self.num_epochs))
        print("Accuracy of the model on the test images: {} ".format(val_acc))
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
        # ---------------------------------------------------------
        if self.enable_lat:
            epsilon, alpha, pro_num = self.set_anp(epoch)
            self.model.update_anp(epsilon, alpha, pro_num)
            # ----------------------------------------------------------

        self.model.train()
        for step, (x, y) in enumerate(train_loader):
            # enable anp training
            if not self.enable_lat:
                pro_num = 1
            if not len(y) == self.batch_size:
                continue
            b_x = Variable(x).to(device)
            b_y = Variable(y).to(device)
            # enable anp training
            if self.enable_lat:
                self.model.zero_reg()
                # progressive process
            for iter in range(pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = self.model(iter_input_x)
                loss = self.criterion(logits, b_y)
                self.optimizer.zero_grad()
                loss.backward()
                # print(model.z0.grad.size())
                # nn.utils.clip_grad_norm_(model.parameters(),args.batchsize)
                self.optimizer.step()
                self.model.save_grad()

            # print train data loss and accuracy
            if step % 20 == 0:
                if self.enable_lat:
                    self.model.zero_reg()

                with torch.no_grad():
                    test_output = self.model(b_x)
                train_loss = self.criterion(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()

                Accuracy = float(
                    (pred_y == b_y.data.cpu().numpy()).astype(int).sum()
                ) / float(b_y.size(0))
                print(
                    "\nTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  \ttrain loss={:.4f} \ttrain accuracy={:.4f}===> ".format(
                        epoch + 1,
                        step + 1,
                        len(train_loader),
                        train_loss.item(),
                        Accuracy,
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

            self.scheduler.step()

            self.train(train_loader, epoch)
            val_acc = self.valid(valid_loader, epoch)

            # self.model.save(name=defense_enhanced_saver)

            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()
                self.model.save(name=defense_enhanced_saver)
            else:
                print(
                    "Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n".format(
                        epoch + 1, best_val_acc
                    )
                )
        return best_model_weights, best_val_acc
