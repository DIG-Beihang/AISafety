#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Yin Zixin
@LastEditors: Yin Zixin
@Description: 
@Date: 2022-1-13
@LastEditTime: 2022-1-13
"""
from ast import Is
import torch
import torchvision
import torch.nn as nn
import os
import PIL
from PIL import Image
from torchvision import transforms, models
import numpy as np
import torch
from torch.autograd import Variable
import cv2
from EvalBox.Attack.AdvAttack.attack import Attack
import torch.nn.functional as F
from functools import reduce


class DualAttentionAttack(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, conv_layer='layer3', **kwargs):
        super(DualAttentionAttack, self).__init__(model, device, IsTargeted)
        # model = self.model
        # print(model)
        self.grad_cam = GradCam(model, conv_layer=conv_layer, device=device)
        self._parse_params(**kwargs)
    
    def _parse_params(self, **kwargs):
        """
        @description:
        @param {
            patchsize: 
            iter: 
            lr: learning rate
        }
        @return: None
        """
        self.patchsize = int(kwargs.get("patchsize", "10"))
        self.iter = int(kwargs.get("iter", "500"))
        self.lr = float(kwargs.get("lr", "0.03"))

    
    def generate(self, xs=None, ys=None):
        """
        @description:
        @param {
            xs:
            ys:
        }
        @return: adv_xs
        """
        device = self.device
        
        adv_xs = []
        batchsize = xs.shape[0]
        for idx in range(batchsize):
            x = xs[idx:idx+1].to(device)
            y = ys[idx].to(device)
            # print(x.shape)
            # print(y.shape)

            patch = torch.randn_like(x).to(device) * 0.01
            mask = torch.zeros((x.shape[2], x.shape[3])).to(device)
            centerx = x.shape[2] // 2
            centery = x.shape[3] // 2
            mask[centerx-self.patchsize//2:centerx+self.patchsize//2, centery-self.patchsize//2:centery+self.patchsize//2] = 1
            patch = patch * mask
            patch = torch.autograd.Variable(patch, requires_grad=True)

            # print(mask)

            optimizer = torch.optim.Adam([patch], lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 7, 15, 25, 60, 120], gamma=1/3)

            for _ in range(self.iter):
                x_input = x * (1-mask) + patch * mask
                # print(x_input)

                x_input = x_input.requires_grad_(True)
                cam, cam_np, pred = self.grad_cam(x_input, y)
                # print(pred)

                if _ > 50 and pred < 0.001:
                    break

                loss = self.loss_midu(cam)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                patch.data.clamp_(-1, 1)
                
                # save_image(x_input, 'x_adv.png')

                scheduler.step()
            adv_xs.append(x * (1-mask) + patch*mask)
        adv_xs = torch.cat(adv_xs, dim=0)
        # print(adv_xs.shape)
        return adv_xs

    def dfs(self, x1, x, y, points, cam_edge):
        points.append(x1[x][y])
        
        self.vis[x][y] = 1
        n = 1
        # print(x, y)
        if x+1 < cam_edge and x1[x+1][y] > 0 and not  self.vis[x+1][y]:
            n += self.dfs(x1, x+1, y, points, cam_edge)
        if x-1 >= 0 and x1[x-1][y] > 0 and not  self.vis[x-1][y]:
            n += self.dfs(x1, x-1, y, points, cam_edge)
        if y+1 < cam_edge and x1[x][y+1] > 0 and not  self.vis[x][y+1]:
            n += self.dfs(x1, x, y+1, points, cam_edge)
        if y-1 >= 0 and x1[x][y-1] > 0 and not  self.vis[x][y-1]:
            n += self.dfs(x1, x, y-1, points, cam_edge)
        return n

    def loss_midu(self, x1):
        # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())
        
        x1 = torch.tanh(x1)
        cam_edge = x1.shape[-1]
        
        self.vis = np.zeros((cam_edge, cam_edge))
        
        loss = []
        # print(x1)
        for i in range(cam_edge):
            for j in range(cam_edge):
                if x1[i][j] > 0 and not self.vis[i][j]:
                    point = []
                    n = self.dfs(x1, i, j, point, cam_edge)
                    # print(n)
                    # print(point)
                    loss.append( reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n) )
        # print(vis)
        if len(loss) == 0:
            return torch.zeros(1).cuda()
        return reduce(lambda x, y: x + y, loss) / len(loss)


# def save_image(image_tensor, save_file):
#     image_tensor = image_tensor.clone()
#     # image_tensor = image_tensor[:16]
#     image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * 0.229 + 0.485 
#     image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * 0.224 + 0.456
#     image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * 0.225 + 0.406
#     torchvision.utils.save_image(image_tensor, save_file, nrow=1)


class GradCam():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, device):
        
        self.model = model.eval()
        self.model.to(device)
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        # class_idx = torch.argmax(scores, axis=-1)
        pred = F.softmax(scores)[0, class_idx]
        # print(class_idx, pred)
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class CAM:
    
    def __init__(self):
        model = models.resnet50(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        self.log_dir = "./"
        
    def __call__(self, img, index, log_dir, t_index=None):
        self.log_dir = log_dir
        self.t_index = t_index
        img = img / 255
        raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        input = self.preprocess_image(img)
        target_index = [468,511,609,817,581,751,627]
        if t_index==None:
            ret, mask, pred = self.grad_cam(input, target_index[index % len(target_index)])
        else:
            ret, mask, pred = self.grad_cam(input, t_index)
        # print(img.shape)
        self.show_cam_on_image(raw_img, mask)
        return ret, pred
        
    def preprocess_image(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        # preprocessed_img = \
        #     np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        # preprocessed_img = torch.from_numpy(preprocessed_img)
        # preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        if self.t_index==None:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
            Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
            
        else:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam_'+str(self.t_index)+'.jpg'))
        Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))