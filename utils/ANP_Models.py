from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from Models.basic_module import BasicModule


class ANP_VGG16(BasicModule):
    def __init__(
        self,
        enable_lat,
        layerlist,
        epsilon,
        alpha,
        pro_num,
        batch_size=128,
        num_classes=10,
        if_dropout=False,
        **kwargs
    ):
        super(ANP_VGG16, self).__init__()
        print("| VGG 16 for CIFAR-10")
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
        self.enable_list = [0 for i in range(14)]  # x , z1~z13
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(",")]
            self.layerlist_digit = [int(x) for x in layerlist.split(",")]
        else:
            self.layerlist = "all"
            self.layerlist_digit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.epsilon = epsilon
        self.alpha = alpha
        self.pro_num = pro_num
        self.if_dropout = if_dropout
        self.choose_layer()
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
        self.attack_step_num = int(kwargs.get("attack_step_num", 40))
        self.num_epochs = int(kwargs.get("num_epochs", 200))
        self.step_size = float(kwargs.get("step_size", 0.01))
        # self.epsilon = float(kwargs.get('epsilon', 0.3))

    def forward(self, x):
        # batch_norm is True for naive VGG
        # x for generating adversarial example

        x.retain_grad()
        self.input = x
        # enable anp training
        if self.enable_lat and self.enable_list[0]:
            self.input.retain_grad()
            # LAT add saved grad to x_reg
            # print('x_reg is {}'.format(self.x_reg.data.size()))
            input_add = self.input.add(self.epsilon / self.pro_num * self.x_reg.data)
        else:
            input_add = self.input
        self.z1 = self.features[0](input_add)  # conv1
        # enable anp training
        if self.enable_lat and self.enable_list[1]:
            self.z1.retain_grad()
            # LAT add saved grad to z1_reg
            z1_add = self.z1.add(self.epsilon / self.pro_num * self.z1_reg.data)
        else:
            z1_add = self.z1
        a1 = self.features[2](self.features[1](z1_add))  # bn1,relu

        self.z2 = self.features[3](a1)  # conv2
        # enable anp training
        if self.enable_lat and self.enable_list[2]:
            self.z2.retain_grad()
            # LAT add saved grad to z2_reg
            z2_add = self.z2.add(self.epsilon / self.pro_num * self.z2_reg.data)
        else:
            z2_add = self.z2
        a2 = self.features[5](self.features[4](z2_add))  # bn2,relu

        p2 = self.features[6](a2)  # maxpooling

        self.z3 = self.features[7](p2)  # conv3
        # enable anp training
        if self.enable_lat and self.enable_list[3]:
            self.z3.retain_grad()
            # LAT add saved grad to z3_reg
            z3_add = self.z3.add(self.epsilon / self.pro_num * self.z3_reg.data)
        else:
            z3_add = self.z3
        a3 = self.features[9](self.features[8](z3_add))  # bn3,relu

        self.z4 = self.features[10](a3)  # conv4
        # enable anp training
        if self.enable_lat and self.enable_list[4]:
            self.z4.retain_grad()
            # LAT add saved grad to z4_reg
            z4_add = self.z4.add(self.epsilon / self.pro_num * self.z4_reg.data)
        else:
            z4_add = self.z4
        a4 = self.features[12](self.features[11](z4_add))  # bn2,relu

        p4 = self.features[13](a4)  # maxpooling

        self.z5 = self.features[14](p4)  # conv5
        # enable anp training
        if self.enable_lat and self.enable_list[5]:
            self.z5.retain_grad()
            # LAT add saved grad to z5_reg
            z5_add = self.z5.add(self.epsilon / self.pro_num * self.z5_reg.data)
        else:
            z5_add = self.z5
        a5 = self.features[16](self.features[15](z5_add))  # bn5,relu

        self.z6 = self.features[17](a5)  # conv6
        # enable anp training
        if self.enable_lat and self.enable_list[6]:
            self.z6.retain_grad()
            # LAT add saved grad to z6_reg
            z6_add = self.z6.add(self.epsilon / self.pro_num * self.z6_reg.data)
        else:
            z6_add = self.z6
        a6 = self.features[19](self.features[18](z6_add))  # bn6,relu

        self.z7 = self.features[20](a6)  # conv7
        # enable anp training
        if self.enable_lat and self.enable_list[7]:
            self.z7.retain_grad()
            # LAT add saved grad to z7_reg
            z7_add = self.z7.add(self.epsilon / self.pro_num * self.z7_reg.data)
        else:
            z7_add = self.z7
        a7 = self.features[22](self.features[21](z7_add))  # bn7,relu

        p7 = self.features[23](a7)  # maxpooling

        self.z8 = self.features[24](p7)  # conv8
        # enable anp training
        if self.enable_lat and self.enable_list[8]:
            self.z8.retain_grad()
            # LAT add saved grad to z8_reg
            z8_add = self.z8.add(self.epsilon / self.pro_num * self.z8_reg.data)
        else:
            z8_add = self.z8
        a8 = self.features[26](self.features[25](z8_add))  # bn8,relu

        self.z9 = self.features[27](a8)  # conv9
        # enable anp training
        if self.enable_lat and self.enable_list[9]:
            self.z9.retain_grad()
            # LAT add saved grad to z9_reg
            z9_add = self.z9.add(self.epsilon / self.pro_num * self.z9_reg.data)
        else:
            z9_add = self.z9
        a9 = self.features[29](self.features[28](z9_add))  # bn9,relu

        self.z10 = self.features[30](a9)  # conv10
        # enable anp training
        if self.enable_lat and self.enable_list[10]:
            self.z10.retain_grad()
            # LAT add saved grad to z10_reg
            z10_add = self.z10.add(self.epsilon / self.pro_num * self.z10_reg.data)
        else:
            z10_add = self.z10
        a10 = self.features[32](self.features[31](z10_add))  # bn10,relu

        p10 = self.features[33](a10)  # maxpooling

        self.z11 = self.features[34](p10)  # conv11
        # enable anp training
        if self.enable_lat and self.enable_list[11]:
            self.z11.retain_grad()
            # LAT add saved grad to z11_reg
            z11_add = self.z11.add(self.epsilon / self.pro_num * self.z11_reg.data)
        else:
            z11_add = self.z11
        a11 = self.features[36](self.features[35](z11_add))  # bn11,relu

        self.z12 = self.features[37](a11)  # conv12
        # enable anp training
        if self.enable_lat and self.enable_list[12]:
            self.z12.retain_grad()
            # LAT add saved grad to z12_reg
            z12_add = self.z12.add(self.epsilon / self.pro_num * self.z12_reg.data)
        else:
            z12_add = self.z12
        a12 = self.features[39](self.features[38](z12_add))  # bn12,relu

        self.z13 = self.features[40](a12)  # conv13
        # enable anp training
        if self.enable_lat and self.enable_list[13]:
            self.z13.retain_grad()
            # LAT add saved grad to z12_reg
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
        for i in range(1, 14):  # z1~z13
            exec("self.z{}_reg.data = self.z{}_reg.data.fill_(0.0)".format(i, i))

    def choose_layer(self):
        if self.enable_lat == False:
            return
        if self.layerlist == "all":
            self.enable_list = [
                1 for i in range(14)
            ]  # all True [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            for i in self.layerlist_digit:
                self.enable_list[i] = 1  # layer list [...,i,...] is True

    def save_grad(self):
        if self.enable_lat:
            for i in self.layerlist_digit:  # [0,1,4,7]
                if i != 0:
                    exec(
                        "self.z{}_reg.data = self.alpha * self.z{}_reg.data + self.z{}.grad / cal_lp_norm(self.z{}.grad,p=2,dim_count=len(self.z{}.grad.size()))".format(
                            i, i, i, i, i
                        )
                    )
                else:
                    self.x_reg.data = (
                        self.alpha * self.x_reg.data
                        + self.input.grad
                        / cal_lp_norm(
                            self.input.grad, p=2, dim_count=len(self.input.grad.size())
                        )
                    )

    def update_anp(self, epsilon, alpha, pro_num):
        self.epsilon = epsilon
        self.alpha = alpha
        self.pro_num = pro_num


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def cal_lp_norm(tensor, p, dim_count):
    tmp = tensor
    for i in range(1, dim_count):
        tmp = torch.norm(tmp, p=p, dim=i, keepdim=True)
    return tmp
