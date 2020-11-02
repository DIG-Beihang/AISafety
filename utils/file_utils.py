import numpy as np
import os
from PIL import Image
import random

from xml.dom.minidom import parse
import xml.dom.minidom

import importlib

# this is the tool function when deal with the imagefile or
# help showing the result


def unpickle(file):  # 该函数将cifar10提供的文件读取到python的数据结构(字典)中
    import pickle

    fo = open(file, "rb")
    dict = pickle.load(fo, encoding="iso-8859-1")
    fo.close()
    return dict


def generateCifar10_dict(path):
    # dict={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

    dict = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck",
    }
    f = open(path, "w")
    f.write(str(dict))
    f.close()
    print("save dict successfully.")


def read_dict_from_file(path):
    # read from local
    f = open(path, "r")
    dict_ = eval(f.read())
    f.close()
    return dict_


# only for dict to the image, like ImageNet
def file_map(label_data, image_name, dictionary):
    return


def file_extension(path):
    return os.path.splitext(path)[1]


# load the data
def random_choose_data(label_path, random_ratio, division):
    sample_names = []
    labels = []
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    item_numbers = len(lines)
    # print("item_numbers",item_numbers)
    # select part of the whole samples
    selected_number = int(item_numbers * random_ratio)

    slice_initial = random.sample(
        lines, selected_number
    )  # if don't change this ,it will be all the same
    # print(len(slice_initial))
    random.shuffle(slice_initial)
    for i in range(len(slice_initial)):
        item = str(slice_initial[i])
        sample_name, label = item.split(division)[0], int(item.split(division)[1])
        sample_names.append(sample_name)
        labels.append(label)

    return sample_names, labels  # output the list and delvery it into ImageFolder


def inorder_choose_data(label_path, random_ratio, division):
    sample_names = []
    labels = []
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    item_numbers = len(lines)
    selected_number = int(item_numbers * random_ratio)
    slice_initial = random.sample(
        lines, selected_number
    )  # if don't change this ,it will be all the same
    random.shuffle(slice_initial)
    for i in range(len(lines)):
        item = str(lines[i])
        sample_name, label = item.split(division)[0], int(item.split(division)[1])
        sample_names.append(sample_name)
        labels.append(label)

    return sample_names, labels  # output the list and delvery it into ImageFolder


# def my data loader, return the data and corresponding label
def default_loader(path):
    return Image.open(path).convert("RGB")  # operation object is the PIL image


def ToPILImage(root_path, image_name):
    return 0


def load_data(data_type, image_path, label_path, ratio):

    if data_type == "ImageNet":
        # sample_names, labels=random_choose_data(label_path,ratio, division=" ")
        sample_names, labels = inorder_choose_data(label_path, ratio, division=" ")
        return sample_names, labels

    elif data_type == "cifar10":
        i = 1
    elif data_type == "cifar100":
        i = 0
    else:
        sample_names, labels = random_choose_data(label_path, ratio, division=" ")
        return sample_names, labels

    return 0


def image_convert_to_numpy():
    return


def load_image_from_path(root_path, extension_filter, max_numbers):
    images = []
    width = 224
    height = 224
    list = os.listdir(root_path)  # 列出文件夹下所有的目录与文件
    length = len(list)
    if length > max_numbers:
        length = max_numbers
    # images = np.zeros((length,width,height,3),dtype=np.uint8)
    for i in range(0, length):
        path = os.path.join(root_path, list[i])
        if os.path.isfile(path):
            if file_extension(path) in extension_filter:
                img = np.array(Image.open(path))
                img = Image.fromarray(img, mode="RGB")
                img = img.resize((width, height), Image.ANTIALIAS)
                images.append(img)

    return images


def show_attack_result(
    origin_sample, adv_sample, origin_outputs, adv_outputs, dictionary
):
    return


def xmlparser(file_path):
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(file_path)
    method = DOMTree.documentElement
    if method.hasAttribute("type"):
        print(method.getAttribute("type"))
    else:
        print("invalid file")
    params = method.getElementsByTagName("param")
    dict = {}
    for param in params:
        if param.hasAttribute("title"):
            contentname = param.getAttribute("title")
            # print(contentname)
            dict[contentname] = (
                param.getElementsByTagName(contentname)[0].childNodes[0].data
            )
            # print(dict)
    return dict


def get_user_model(path):
    # print("path = ", path)
    # 从用户上传路径获取用户模型
    module_user = importlib.import_module(path)
    model = module_user.getModel()
    return model


def get_user_model_defense(path, **kwargs):
    # print("path = ", path)
    # 从用户上传路径获取用户模型
    module_user = importlib.import_module(path)
    model = module_user.getModel_defense(**kwargs)
    return model


def get_user_model_origin(path):
    # print("path = ", path)
    # 从用户上传路径获取用户模型
    module_user = importlib.import_module(path)
    model = module_user.getModel()
    return model
