import torch
import torchvision
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms, models


# pretrained=True
def AlexNet_modify(pretrained=True):
    model = models.alexnet(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 200),
    )
    # model.classifier = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, 200),
    # )

    return model


def getModel():
    return AlexNet_modify()
