import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import PIL
from PIL import Image
from torchvision import transforms, models
import numpy as np


def preprocess(image, resize, device):
    # print(resize)
    trans = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor(),]
    )
    return trans(image.cpu()).to(device)


def save_patched_pic(adv_image, path):
    transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),])
    adv_image = transform(adv_image)
    adv_image.save(path, quality=100, sub_sampling=0)
