#!/usr/bin/env python
# coding=UTF-8
"""
@Author: 
@LastEditors: 
@Description: 
@Date: 2019-06-18 15:09
@LastEditTime: 2019-06-19
"""
#!/usr/bin/env python
# coding=UTF-8
"""
@Author: 
@LastEditors: 
@Description: 
@Date: 2019-06-19 
@LastEditTime: 2019-06-19 
"""
import numpy as np
import os
import torch
from torch.autograd import Variable
from skimage.transform import rescale
from utils.Defense_utils import adjust_learning_rate
from EvalBox.Defense.defense import Defense


class RAND(Defense):
    def __init__(
        self, model=None, device=None, optimizer=None, scheduler=None, **kwargs
    ):
        """
        @description: Input Randomization (Rand)
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
        self.dataset = kwargs.get("dataset", "CIFAR10")
        if self.dataset == "CIFAR10":
            self.resize = kwargs.get("resize", 36)
        elif self.dataset == "ImageNet":
            self.resize = kwargs.get("resize", 256)
        self.num_epochs = int(kwargs.get("num_epochs", 200))

    def randomization_transformation(
        self, samples=None, original_size=None, final_size=None
    ):
        """

        :param samples:
        :param original_size:
        :param final_size:
        :return:
        """
        device = self.device
        self.model.eval().to(device)
        # Convert torch Tensor to numpy array
        if torch.is_tensor(samples) is True:
            samples = samples.cpu().numpy()
        # convert the channel of images
        samples = np.transpose(samples, (0, 2, 3, 1))
        assert (
            samples.shape[-1] == 1 or samples.shape[-1] == 3
        ), "in the randomization transform function, channel must be placed in the last"

        transformed_samples = []
        # print ('transforming the images (size: {}) ...'.format(samples.shape))
        for image in samples:
            # Step 1: Random Resizing Layer
            # specify the random size which the image will be rescaled to
            rnd = np.random.randint(
                original_size, final_size
            )  # original_size和final_size之间的一个随机数
            scale = (rnd * 1.0) / original_size
            rescaled_image = rescale(
                image=image,
                scale=scale,
                multichannel=True,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )

            # Step 2: Random Padding Layer
            h_rem = final_size - rnd
            w_rem = final_size - rnd
            pad_left = np.random.randint(0, w_rem)
            pad_right = w_rem - pad_left
            pad_top = np.random.randint(0, h_rem)
            pad_bottom = h_rem - pad_top
            # padding the image to the new size using gray pixels
            padded_image = np.pad(
                rescaled_image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                "constant",
                constant_values=0.5,
            )
            transformed_samples.append(padded_image)

        # reset the channel location back and convert numpy back as the Tensor
        transformed_samples = np.array(transformed_samples)
        transformed_samples = (
            torch.from_numpy(np.transpose(transformed_samples, (0, 3, 1, 2)))
            .float()
            .to(self.device)
        )
        return transformed_samples

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
            rand_images = images.to(device)
            rand_labels = labels.to(device)

            self.model.eval()
            # input images first go through the randomization transformation layer and then the resulting images are feed into the original model
            adv_images = self.randomization_transformation(
                samples=images, original_size=images.shape[-1], final_size=self.resize
            )
            adv_images = adv_images.to(self.device)

            self.model.train()

            logits_rand = self.model(rand_images)
            loss_rand = self.criterion(logits_rand, rand_labels)

            logits_adv = self.model(adv_images)
            loss_adv = self.criterion(logits_adv, rand_labels)

            loss = 0.5 * (loss_rand + loss_adv)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                "\rTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  \tloss_rand={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> ".format(
                    epoch,
                    index,
                    len(train_loader),
                    loss_rand.item(),
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
