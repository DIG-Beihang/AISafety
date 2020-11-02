#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/12 14:14
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : basic_module.py
# **************************************

import torch
import torchvision.models as models


class BasicModule(torch.nn.Module):
    """
    encapsulate the nn.Module to providing both load and save functions
    """

    def __init__(self, **kwargs):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        raise NotImplementedError

    # load the model
    def load(self, path, device):
        """

        :param path:
        :param device:
        :return:
        """

        print(
            "starting to LOAD the ${}$ Model from {} within the {} device".format(
                self.model_name, path, device
            )
        )
        if device == torch.device("cpu"):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            # original=torch.load(path, map_location='cpu')
        #            new = {"model": original["model"]}

        # checkpoint = torch.load(path, map_location='cpu')
        # net=models.resnet18()
        # net = torch.nn.DataParallel(net)
        # net.load_state_dict()
        else:
            self.load_state_dict(torch.load(path))

    # save the model
    def save(self, name=None):
        """

        :param name:
        :return:
        """
        assert name is not None, "please specify the path name to save the module"

        with open(name, "wb") as file:
            torch.save(self.state_dict(), file)
        print("starting to SAVE the ${}$ Model to ${}$\n".format(self.model_name, name))
