#!/usr/bin/env python
# coding=UTF-8
"""
@Author: linna
@LastEditors: linna
@Description: 
@Date: 2020-05-08 14:50:02
@LastEditTime: 2020-05-09 09:11:02
"""
import numpy as np
import torch
from PIL import Image

from EvalBox.Attack.CorAttack.corruptions import *


class CORRUPT:
    def __init__(self, severity=1, corruption_name=None, **kwargs):
        """
        @description: 
        @param {
            severity:
            corruption_name:(gaussian_noise, shot_noise, impulse_noise,
                            defocus_blur, glass_blur, motion_blur, zoom_blur,
                            snow, frost, fog, brightness, contrast,
                            elastic_transform, pixelate, jpeg_compression,
                            speckle_noise, gaussian_blur, spatter, saturate)
        } 
        @return: None
        """

        self.severity = severity
        self.corruption_name = corruption_name
        self._parse_params(**kwargs)

        corruption_tuple = (
            gaussian_noise,
            shot_noise,
            impulse_noise,
            defocus_blur,
            glass_blur,
            motion_blur,
            zoom_blur,
            snow,
            frost,
            fog,
            brightness,
            contrast,
            elastic_transform,
            pixelate,
            jpeg_compression,
            speckle_noise,
            gaussian_blur,
            spatter,
            saturate,
        )

        corruption_dict = {
            corr_func.__name__: corr_func for corr_func in corruption_tuple
        }
        if corruption_name not in corruption_dict:
            print("Error corruption name!!!")
            raise NotImplementedError
        self.cor_attck = corruption_dict[corruption_name]

    def _parse_params(self, **kwargs):
        """
        @description:
        @param {
            epsilon:
        }
        @return: None
        """
        # 沿着梯度方向步长的参数
        self.corruption_name = kwargs.get("corruption_name", "gaussian_noise")
        self.severity = int(kwargs.get("severity", 1))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
            device:
        } 
        @return: 
        """
        copy_xs = xs.permute(0, 2, 3, 1).numpy()
        cor_xs = []
        for x in copy_xs:
            x = (x * 255).astype(np.uint8)
            cor_x = self.cor_attck(Image.fromarray(x * 255), self.severity)
            cor_xs.append(torch.from_numpy(cor_x / 255.0))

        cor_xs = torch.stack(cor_xs, 0).permute(0, 3, 1, 2)
        cor_xs = torch.tensor(cor_xs, dtype=torch.float32)
        # print(cor_xs.shape)
        return cor_xs
