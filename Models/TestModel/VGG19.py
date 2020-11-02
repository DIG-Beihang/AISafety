import torch
import torchvision
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms, models


# pretrained=True
def VGG19(pretrained=True):
    model = models.vgg19(pretrained=False)

    return model


def getModel():
    return VGG19()
