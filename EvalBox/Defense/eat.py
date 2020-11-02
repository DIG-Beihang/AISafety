#!/usr/bin/env python
# coding=UTF-8
"""
@Author:
@LastEditors:
@Description: 
@Date: 2019-6-17 10:26:12
@LastEditTime: 2019-6-18
"""
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.EAT_External_Models import CIFAR10_A, CIFAR10_B, CIFAR10_C, CIFAR10_D
from utils.EAT_External_Models import ImageNet_A, ImageNet_B, ImageNet_C, ImageNet_D
from utils.Defense_utils import adjust_learning_rate
from EvalBox.Defense.defense import Defense


class EAT(Defense):
    def __init__(
        self, model=None, device=None, optimizer=None, scheduler=None, **kwargs
    ):
        """
        @description: Ensemble adversarial training (EAT)
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
        self.dataset = str(kwargs.get("dataset"))
        self.num_epochs = int(kwargs.get("num_epochs", 100))
        self.epsilon = float(kwargs.get("epsilon", 0.3))
        self.alpha = float(kwargs.get("alpha", 0.05))
        self.train_externals = kwargs.get("train_externals", True)
        self.learn_rate = float(kwargs.get("learn_rate", 0.001))
        self.external_model_path = str(
            kwargs.get("external_model_path", "../EvalBox/DefenseEnhancedModels/EAT/")
        )

    def train_one_epoch(self, model, train_loader, optimizer, epoch, device):
        """

        :param model:
        :param train_loader:
        :param optimizer:
        :param epoch:
        :param device:
        :return:
        """

        # Sets the model in training mode
        model.train()
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward the nn
            outputs = model(images)
            loss = self.criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> ".format(
                    epoch,
                    index,
                    len(train_loader),
                    index / len(train_loader) * 100.0,
                    loss.item(),
                ),
                end=" ",
            )

    def train_external_model_group(
        self, train_loader=None, validation_loader=None, model_dir_path=None
    ):
        """

        :param train_loader:
        :param validation_loader:
        :return:
        """
        # Set up the model group with 4 static external models

        if self.dataset == "CIFAR10":
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        elif self.dataset == "ImageNet":
            model_group = [ImageNet_A(), ImageNet_B(), ImageNet_C(), ImageNet_D()]

        model_group = [model.to(self.device) for model in model_group]

        # training the models in model_group one by one
        for i in range(len(model_group)):

            # prepare the optimizer for CIFAR10
            if i == 3:
                optimizer_external = optim.SGD(
                    model_group[i].parameters(),
                    lr=0.001,
                    momentum=0.9,
                    weight_decay=1e-6,
                )
            else:
                optimizer_external = optim.Adam(
                    model_group[i].parameters(), lr=self.learn_rate
                )

            # scheduler_external = optim.lr_scheduler.StepLR(optimizer_external, 20, gamma=0.1)

            print("\nwe are training the {}-th static external model ......".format(i))
            best_val_acc = None
            for index_epoch in range(self.num_epochs):

                # scheduler_external.step()
                # print("external model learn rate is: ", scheduler_external.get_lr()[0])

                self.train_one_epoch(
                    model=model_group[i],
                    train_loader=train_loader,
                    optimizer=optimizer_external,
                    epoch=index_epoch,
                    device=self.device,
                )
                val_acc = self.valid(
                    model=model_group[i], valid_loader=validation_loader
                )

                adjust_learning_rate(epoch=index_epoch, optimizer=optimizer_external)
                # print(model_dir_path)
                assert os.path.exists(model_dir_path)
                defense_external_saver = os.path.join(
                    model_dir_path, "{}_EAT_{}.pt".format(self.dataset, str(i))
                )
                if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                    if best_val_acc is not None:
                        os.remove(defense_external_saver)
                    best_val_acc = val_acc
                    model_group[i].save(name=defense_external_saver)
                else:
                    print(
                        "Train Epoch {:>3}: validation dataset accuracy did not improve from {:.4f}\n".format(
                            index_epoch, best_val_acc
                        )
                    )

    def load_external_model_group(self, model_dir=None):
        self.return_ = """"
        :param model_dir:
        :param test_loader:
        :return:
        """
        print("\n!!! Loading static external models ...")
        # Set up 4 static external models
        # print(self.dataset)
        model_group = None
        if self.dataset == "CIFAR10":
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        elif self.dataset == "ImageNet":
            model_group = [ImageNet_A(), ImageNet_B(), ImageNet_C(), ImageNet_D()]

        model_group = [model.to(self.device) for model in model_group]

        for i in range(len(model_group)):
            print("loading the {}-th static external model".format(i))
            model_path = "{}/{}_EAT_{}.pt".format(model_dir, self.dataset, str(i))
            assert os.path.exists(
                model_path
            ), "please train the external model first!!!"
            model_group[i].load(path=model_path, device=self.device)

        return model_group

    def random_fgsm_generation(self, model=None, natural_images=None):
        """
         A new randomized single step attack (RFGSM)
         :param model:
         :param natural_images:
         :return:
         """
        attack_model = model.to(self.device)
        attack_model.eval()

        with torch.no_grad():
            random_sign = torch.sign(torch.randn(*natural_images.size())).to(
                self.device
            )
            new_images = torch.clamp(
                natural_images + self.alpha * random_sign, min=0.0, max=1.0
            )

        new_images.requires_grad = True

        logits_attack = attack_model(new_images)
        # To avoid label leaking, we use the model's output instead of the true labels
        labels_attack = torch.max(logits_attack, dim=1)[1]
        loss_attack = self.criterion(logits_attack, labels_attack)
        gradient = torch.autograd.grad(loss_attack, new_images)[0]

        new_images.requires_grad = False

        # generation of adversarial examples
        with torch.no_grad():
            xs_adv = new_images + (self.epsilon - self.alpha) * torch.sign(gradient)
            xs_adv = torch.clamp(xs_adv, min=0.0, max=1.0)
        return xs_adv

    def valid(self, model=None, valid_loader=None):
        """
        @description:
        @param {
            valid_loader:
            epoch:
        }
        @return: val_acc
        """
        device = self.device
        model.to(device).eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
            val_acc = correct / total
        return val_acc

    def train(self, pre_trained_models, train_loader=None, epoch=None):
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
            eat_images = images.to(device)
            eat_labels = labels.to(device)

            self.model.eval()
            # in each mini_batch, we randomly choose the attack model which adversarial examples are generated on
            idx = np.random.randint(5)
            if idx == 0:
                attacking_model = self.model
            else:
                attacking_model = pre_trained_models[idx - 1]

            adv_images = self.random_fgsm_generation(
                model=attacking_model, natural_images=eat_images
            )

            self.model.train()

            logits_eat = self.model(eat_images)
            loss_eat = self.criterion(logits_eat, eat_labels)

            logits_adv = self.model(adv_images)
            loss_adv = self.criterion(logits_adv, eat_labels)

            loss = 0.5 * (loss_eat + loss_adv)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                "\rTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  \tloss_eat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> ".format(
                    epoch,
                    index,
                    len(train_loader),
                    loss_eat.item(),
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
        dir_path = os.path.dirname(defense_enhanced_saver)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # print(self.train_externals)
        if self.train_externals:
            print("\nStart to train the external models ......\n")
            self.train_external_model_group(
                train_loader=train_loader,
                validation_loader=valid_loader,
                model_dir_path=self.external_model_path,
            )

            # load the external models
        pre_train_models = self.load_external_model_group(
            model_dir=self.external_model_path
        )

        best_val_acc = None
        best_model_weights = self.model.state_dict()

        for epoch in range(self.num_epochs):
            # if not self.scheduler:
            # self.scheduler.step()

            self.train(pre_train_models, train_loader, epoch)
            val_acc = self.valid(self.model, valid_loader)

            adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)

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

            # if round(val_acc, 4) >= round(best_acc, 4):
            #     best_acc = val_acc
            #     best_model_weights = self.model.state_dict()

        print("Best val Acc: {:.4f}".format(best_val_acc))
        return best_model_weights, best_val_acc
