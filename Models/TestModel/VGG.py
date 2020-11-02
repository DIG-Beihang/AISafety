import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model


class VGG16(nn.Module):
    def __init__(
        self,
        enable_lat,
        epsilon,
        pro_num,
        batch_size=128,
        num_classes=10,
        if_dropout=False,
    ):

        super(VGG16, self).__init__()
        self.batch_size = batch_size
        self.cf_dict = [
            64,
            64,
            "mp",
            128,
            128,
            "mp",
            256,
            256,
            256,
            "mp",
            512,
            512,
            512,
            "mp",
            512,
            512,
            512,
            "mp",
        ]
        self.z_list = [
            "z1_reg",
            "z2_reg",
            "z3_reg",
            "z4_reg",
            "z5_reg",
            "z6_reg",
            "z7_reg",
            "z8_reg",
            "z9_reg",
            "z10_reg",
            "z11_reg",
            "z12_reg",
            "z13_reg",
        ]
        self.register_buffer("x_reg", torch.zeros([batch_size, 3, 32, 32]))
        self.reg_size_list = list()
        self.features = self._make_layers()
        self.linear = nn.Linear(512, num_classes)
        self.enable_lat = enable_lat
        self.epsilon = epsilon
        self.pro_num = pro_num
        self.if_dropout = if_dropout
        self.smodel = None

    def load(self, path, device):
        """

        :param path:
        :param device:
        :return:
        """

        #        print('starting to LOAD the ${}$ Model from {} within the {} device'.format(self.model_name, path, device))
        if device == torch.device("cpu"):
            self.model = self.load_state_dict(torch.load(path, map_location="cpu"))

            print("aaa", self.model)
        else:
            self.model = self.load_state_dict(torch.load(path))
            print("aaa", self.model)

        self.device = device

    def forward(self, x):
        # batch_norm is True for naive VGG
        # x for generating adversarial example

        x.retain_grad()
        self.input = x
        if self.enable_lat:
            self.input.retain_grad()
            # LAT add saved grad to x_reg
            input_add = self.input.add(self.epsilon / self.pro_num * self.x_reg.data)
        else:
            input_add = self.input

        self.z1 = self.features[0](input_add)  # conv1
        if self.enable_lat:
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * self.z1_reg.data)
        else:
            z1_add = self.z1
        a1 = self.features[2](self.features[1](z1_add))  # bn1,relu

        self.z2 = self.features[3](a1)  # conv2
        if self.enable_lat:
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * self.z2_reg.data)
        else:
            z2_add = self.z2
        a2 = self.features[5](self.features[4](z2_add))  # bn2,relu

        p2 = self.features[6](a2)  # maxpooling

        self.z3 = self.features[7](p2)  # conv3
        if self.enable_lat:
            self.z3.retain_grad()
            # LAT add saved grad to z3_reg
            z3_add = self.z3.add(self.epsilon / self.pro_num * self.z3_reg.data)
        else:
            z3_add = self.z3
        a3 = self.features[9](self.features[8](z3_add))  # bn3,relu

        self.z4 = self.features[10](a3)  # conv4
        if self.enable_lat:
            self.z4.retain_grad()
            # LAT add saved grad to z4_reg
            z4_add = self.z4.add(self.epsilon / self.pro_num * self.z4_reg.data)
        else:
            z4_add = self.z4
        a4 = self.features[12](self.features[11](z4_add))  # bn2,relu

        p4 = self.features[13](a4)  # maxpooling

        self.z5 = self.features[14](p4)  # conv5
        if self.enable_lat:
            self.z5.retain_grad()
            # LAT add saved grad to z5_reg
            z5_add = self.z5.add(self.epsilon / self.pro_num * self.z5_reg.data)
        else:
            z5_add = self.z5
        a5 = self.features[16](self.features[15](z5_add))  # bn5,relu

        self.z6 = self.features[17](a5)  # conv6
        if self.enable_lat:
            self.z6.retain_grad()
            # LAT add saved grad to z6_reg
            z6_add = self.z6.add(self.epsilon / self.pro_num * self.z6_reg.data)
        else:
            z6_add = self.z6
        a6 = self.features[19](self.features[18](z6_add))  # bn6,relu

        self.z7 = self.features[20](a6)  # conv7
        if self.enable_lat:
            self.z7.retain_grad()
            # LAT add saved grad to z7_reg
            z7_add = self.z7.add(self.epsilon / self.pro_num * self.z7_reg.data)
        else:
            z7_add = self.z7
        a7 = self.features[22](self.features[21](z7_add))  # bn7,relu

        p7 = self.features[23](a7)  # maxpooling

        self.z8 = self.features[24](p7)  # conv8
        if self.enable_lat:
            self.z8.retain_grad()
            # LAT add saved grad to z8_reg
            z8_add = self.z8.add(self.epsilon / self.pro_num * self.z8_reg.data)
        else:
            z8_add = self.z8
        a8 = self.features[26](self.features[25](z8_add))  # bn8,relu

        self.z9 = self.features[27](a8)  # conv9
        if self.enable_lat:
            self.z9.retain_grad()
            # LAT add saved grad to z9_reg
            z9_add = self.z9.add(self.epsilon / self.pro_num * self.z9_reg.data)
        else:
            z9_add = self.z9
        a9 = self.features[29](self.features[28](z9_add))  # bn9,relu

        self.z10 = self.features[30](a9)  # conv10
        if self.enable_lat:
            self.z10.retain_grad()
            # LAT add saved grad to z10_reg
            z10_add = self.z10.add(self.epsilon / self.pro_num * self.z10_reg.data)
        else:
            z10_add = self.z10
        a10 = self.features[32](self.features[31](z10_add))  # bn10,relu

        p10 = self.features[33](a10)  # maxpooling

        self.z11 = self.features[34](p10)  # conv11
        if self.enable_lat:
            self.z11.retain_grad()
            # LAT add saved grad to z11_reg
            z11_add = self.z11.add(self.epsilon / self.pro_num * self.z11_reg.data)
        else:
            z11_add = self.z11
        a11 = self.features[36](self.features[35](z11_add))  # bn11,relu

        self.z12 = self.features[37](a11)  # conv12
        if self.enable_lat:
            self.z12.retain_grad()
            # LAT add saved grad to z12_reg
            z12_add = self.z12.add(self.epsilon / self.pro_num * self.z12_reg.data)
        else:
            z12_add = self.z12
        a12 = self.features[39](self.features[38](z12_add))  # bn12,relu

        self.z13 = self.features[40](a12)  # conv13
        if self.enable_lat:
            self.z13.retain_grad()
            # LAT add saved grad to z13_reg
            z13_add = self.z13.add(self.epsilon / self.pro_num * self.z13_reg.data)
        else:
            z13_add = self.z13
        a13 = self.features[42](self.features[41](z13_add))  # bn13,relu

        p13 = self.features[43](a13)  # maxpooling

        out = self.features[44](p13)  # avgpooling
        # out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.if_dropout:
            out = F.dropout(out, p=0.3, training=self.training)
        out = self.linear(out)

        return out

    def predict(self, data):
        #        print(self.model,"var_xs")
        device = torch.device("cpu")
        xs = torch.from_numpy(data)
        print(xs.shape)
        var_xs = Variable(xs.to(device))
        print(var_xs.shape)
        return model(var_xs)

    def convert_tf2torch(self, data):
        device = torch.device("cpu")
        xs = torch.from_numpy(data)
        print(xs.shape)
        var_xs = Variable(xs.to(device))
        return var_xs

    def _make_layers(self):
        layers = []
        in_planes = 3
        imgSize = 32
        z_index = 0
        for x in self.cf_dict:
            if x == "mp":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                imgSize /= 2
            else:
                layers += [
                    conv3x3(in_planes, x),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_planes = x
                self.register_buffer(
                    self.z_list[z_index],
                    torch.zeros(
                        [self.batch_size, in_planes, (int)(imgSize), (int)(imgSize)]
                    ),
                )
                self.reg_size_list.append(
                    [self.batch_size, in_planes, (int)(imgSize), (int)(imgSize)]
                )
                z_index += 1
        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def zero_reg(self):
        self.x_reg.data = self.x_reg.data.fill_(0.0)
        self.z1_reg.data = self.z1_reg.data.fill_(0.0)
        self.z2_reg.data = self.z2_reg.data.fill_(0.0)
        self.z3_reg.data = self.z3_reg.data.fill_(0.0)
        self.z4_reg.data = self.z4_reg.data.fill_(0.0)
        self.z5_reg.data = self.z5_reg.data.fill_(0.0)
        self.z6_reg.data = self.z6_reg.data.fill_(0.0)
        self.z7_reg.data = self.z7_reg.data.fill_(0.0)
        self.z8_reg.data = self.z8_reg.data.fill_(0.0)
        self.z9_reg.data = self.z9_reg.data.fill_(0.0)
        self.z10_reg.data = self.z10_reg.data.fill_(0.0)
        self.z11_reg.data = self.z11_reg.data.fill_(0.0)
        self.z12_reg.data = self.z12_reg.data.fill_(0.0)
        self.z13_reg.data = self.z13_reg.data.fill_(0.0)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


if __name__ == "__main__":
    net = VGG16(
        enable_lat=True,
        epsilon=0.3,
        pro_num=3,
        batch_size=64,
        num_classes=10,
        if_dropout=False,
    )
    x = Variable(torch.randn(64, 3, 32, 32))
    y = net(x)
    print(x.size())
    print(y.size())
    layers = []
    for i in range(0, 45):
        layers += [net.features[i]]
    a = nn.Sequential(*layers)
    print(a(x).size())
    print(net.z7_reg.size())


def getModel():
    return VGG16(
        enable_lat=True,
        epsilon=0.3,
        pro_num=3,
        batch_size=64,
        num_classes=1000,
        if_dropout=False,
    )
