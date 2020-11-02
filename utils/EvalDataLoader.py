import torch
from torch.utils.data import Dataset
import pandas as pd
import utils.file_utils
from utils.file_utils import load_data, default_loader
import torch.utils.data as Data
from torchvision import datasets, transforms
import os
from PIL.Image import Image
import numpy as np

# 定义自己的类
class EvalDataset(Dataset):
    # 初始化
    def __init__(
        self,
        root_path,
        label_path,
        origin_path=None,
        origin_label_path=None,
        data_type=None,
        image_size=None,
        transform=None,
        ratio=None,
    ):
        # 读入数据

        imgs, labels = load_data(data_type, root_path, label_path, ratio)
        imgs_origin = None
        labels_origin = None
        if not origin_path == None and not origin_label_path == None:
            imgs_origin, labels_origin = load_data(
                data_type, origin_path, origin_label_path, ratio
            )

        self.imgs_origin = imgs_origin
        self.labels_origin = labels_origin
        self.imgs = imgs
        self.labels = labels
        self.loader = default_loader
        self.image_size = image_size
        self.root_path = root_path
        self.origin_path = origin_path
        self.transform = transform

    # 返回df的长度
    def __len__(self):
        return len(self.imgs)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        img = self.loader(os.path.join(self.root_path, self.imgs[idx]))
        img = img.resize((int(self.image_size[0]), int(self.image_size[1])))
        # img = img.resize(self.image_size)
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]

        if not self.imgs_origin == None and not self.labels_origin == None:
            img_origin = self.loader(
                os.path.join(self.origin_path, self.imgs_origin[idx])
            )
            img_origin = img_origin.resize((self.image_size, self.image_size))
            if self.transform is not None:
                img_origin = self.transform(img_origin)
            label_origin = self.labels_origin[idx]
            return img, label, img_origin, label_origin
        else:
            return img, label
