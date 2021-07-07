import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
import torchvision.transforms as transforms
from torchvision import utils
import torch.utils.data as Data
import cv2
import sys
import numpy as np
import argparse
import os
sys.path.append('%s/../../' % os.path.dirname(os.path.realpath(__file__)))
import PIL.Image as Image
import importlib

class NeuronVisualizer:
    def __init__(self, model, model_name, layer_list, img_size):
        self.model = model
        self.model_name = model_name
        self.module_name = []
        self.features_out_hook = []
        self.hook_handle_list = []
        self.layer_list = layer_list
        self.img_size = img_size
        self.segment_size = img_size
        self.margin = 3
        self.max_knum = 5
        self.k_rate = 0.1
        self.threshold_scale = 0.25

        self.output_folder = 'result_segments/%s' % self.model_name
        self.input = None
        self.cln_xs = None

    def _for_hook(self, module, fea_in, fea_out):
        self.features_out_hook.append(np.squeeze(fea_out.data.cpu().numpy()))

    def _handle_hook(self):
        for name, module in self.model._modules.items():
            self.module_name.append(name)
            handle = module.register_forward_hook(self._for_hook)
            self.hook_handle_list.append(handle)

    def _release_hook(self):
        for handle in self.hook_handle_list:
            handle.remove()

    def _predict(self, input):
        with torch.no_grad():
            pred = self.model(input)

    def _save_results(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder + '/image')

    def _calculate_features(self):
        for layer_id, key in enumerate(self.features_out_hook):
            # 输入层与输出层排除
            if "conv" in self.module_name[layer_id].lower() or len(key.shape) < 3:
                continue
            # 若指定layer_list，则仅输出指定层输出
            if len(self.layer_list) > 0 and self.module_name[layer_id].lower() not in self.layer_list:
                continue
            num_units = self.features_out_hook[layer_id].shape[1]
            for inputID in range(len(self.cln_img)):
                # 取出前 k 激活的神经元
                feature_map = self.features_out_hook[layer_id][inputID]
                knum = min(self.max_knum, max(1, int(len(feature_map) * self.k_rate)))
                topk = np.argsort(np.mean(np.mean(feature_map, 1), 1))
                topk = topk[-knum:]
                whole_layer_output = []
                for unitID in range(knum):
                    index = topk[unitID]
                    feature_unit = feature_map[index] / np.max(feature_map[index])
                    # 生成 mask
                    mask = cv2.resize(feature_unit, self.segment_size)
                    mask[mask < self.threshold_scale] = 0.0 # binarize the mask
                    mask[mask > self.threshold_scale] = 1.0

                    img = self.cln_img[inputID].numpy()
                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, self.segment_size)
                    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    img_mask = np.multiply(img, mask[:,:, np.newaxis])
                    img_mask = np.uint8(img_mask * 255)
                    output_unit = []
                    output_unit.append(img_mask)
                    # whole_layer_output.append(img_mask)
                    # whole_layer_output.append(np.uint8(np.ones((self.segment_size[0], self.margin, 3))*255))

                    montage_unit = np.concatenate(output_unit, axis=1)
                    cv2.imwrite(os.path.join(self.output_folder, 'image', 'Pic%d-%s-%d.jpg'%(inputID, self.module_name[layer_id], unitID)), montage_unit)

                # montage_unit = np.concatenate(output_unit, axis=1)
                # cv2.imwrite(os.path.join(self.output_folder, 'image', 'Pic%d-%s-%d.jpg'%(inputID, self.module_name[layer_id], unitID)), montage_unit)
    def __call__(self, cln_img, input):
        # 添加挂锁
        self._handle_hook()

        # 执行预测过程
        self._predict(input)

        # 取消挂锁
        self._release_hook()
        
        self.input = input
        self.cln_img = cln_img

        self._save_results()
        self._calculate_features()

def init_model(model_path = '', pth_path = ''):
    model_lib = importlib.import_module(model_path)
    model = model_lib.getModel()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    return model

if __name__ == '__main__':
    """
    1. Loads an image.
    2. transform it and converts to a pytorch variable.
    3. Makes a forward pass to get the neuron outputs.
    4. Makes the visualization. """
    model = init_model('Models.UserModel.ResNet2', '../../Models/weights/resnet20_cifar10_clean_new.pt')
    model = model.eval()

    visualizer = NeuronVisualizer(model = model, model_name = "ResNet2", layer_list = [], img_size = (32, 32))
    dataset = []
    cln_xs = []
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    for i in range(2):
        im = cv2.imread('../../Datasets/cifar_test/flip_data/' + str(i) + '.jpg')
        im = np.ascontiguousarray(im[:,:,::-1])
        cln_xs.append(im)
        im = transform(im)
        dataset.append(np.array(im))
    cln_xs = np.array(cln_xs)
    cln_xs = torch.from_numpy(cln_xs)
    dataset = np.array(dataset)
    dataset = torch.from_numpy(dataset)
    visualizer(cln_xs, dataset)
