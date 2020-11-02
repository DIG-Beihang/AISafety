import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import PIL
from PIL import Image
from torchvision import transforms, models
import numpy as np


def adjust_learning_rate(epoch, optimizer):
    minimum_learning_rate = 0.5e-6
    for param_group in optimizer.param_groups:
        lr_temp = param_group["lr"]
        if epoch == 80 or epoch == 120 or epoch == 160:
            lr_temp = lr_temp * 1e-1
        elif epoch == 180:
            lr_temp = lr_temp * 5e-1
        param_group["lr"] = max(lr_temp, minimum_learning_rate)
        print(
            "The **learning rate** of the {} epoch is {}".format(
                epoch, param_group["lr"]
            )
        )
