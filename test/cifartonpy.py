from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import random

# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == "2":
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict["data"]  # X, ndarray, 像素值
        Y = datadict["labels"]  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 1, 2, 3).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR100_batch(filename, number):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict["data"]  # X, ndarray, 像素值
        Y = datadict["fine_labels"]  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(number, 3, 32, 32).transpose(0, 1, 2, 3).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def load_CIFAR100(ROOT, typeName="train", numberdata=5000):
    """ load all of cifar """
    f = os.path.join(ROOT, typeName)
    Xtr, Ytr = load_CIFAR100_batch(f, number=numberdata)
    return Xtr, Ytr


def save_numpy(
    X,  # 全部的数据
    Y,  # 全部的数据
    path,  # npy 数据保存的目标路径
    number=10000,  # 最后的要保存的数据的数量，<=总的
    shuff="random_equally",  # 随机选取的类别数量要均衡的方式
    datasetType="cifar10",  # 数据集的名称
    IsTargeted=False,  # 是否生成的是用于目标攻击的标签，随机值（和原始的不同即可)
):

    class_num = 10
    if datasetType == "cifar100":
        class_num = 100
    ys = []
    X_shuffe = np.zeros((number, 3, 32, 32), dtype=np.double)
    Y_shuffe = np.zeros((number, 1), dtype=np.uint8)
    class_number_list = [0 for i in range(class_num)]
    # print(class_number_list)
    # 选择平均数目的不同类别，例如cifar10，选择生成 1000个的话，每一类选择1000/10=100个
    label_one_choice_number = int(number / class_num)

    index = 0
    if shuff == "random_equally":
        # print(Y.shape[0])
        for i in range(Y.shape[0]):
            for j in range(class_num):
                if class_number_list[j] < label_one_choice_number:
                    if Y[i] == j:
                        class_number_list[j] += 1
                        X_shuffe[index] = X[i]
                        Y_shuffe[index] = Y[i]
                        index += 1
    else:
        # 直接保存前 number个
        for i in range(number):
            X_shuffe[index] = X[i]
            Y_shuffe[index] = Y[i]
            index += 1
    # print(class_number_list,Y_shuffe)

    key = np.unique(Y_shuffe)
    result = {}
    for k in key:
        mask = Y_shuffe == k
        y_new = Y_shuffe[mask]
        v = y_new.size
        result[k] = v
    print("check every type is include and in average", result)

    if not IsTargeted:
        for i in range(Y_shuffe.shape[0]):
            y = np.zeros((1, class_num), dtype=np.uint8)
            y[0][Y_shuffe[i]] = 1
            ys.append(y[0])
            # print(y[0])

        np.save(
            path + "{}_{}_origin_labels.npy".format(datasetType, number), np.array(ys)
        )
        print(
            "save the npy file in path :",
            path + "{}_{}_origin_labels.npy".format(datasetType, number),
        )
        np.save(
            path + "{}_{}_origin_inputs.npy".format(datasetType, number),
            np.array(X_shuffe / 255),
        )
        print(
            "save the npy file in path :",
            path + "{}_{}_origin_inputs.npy".format(datasetType, number),
        )
    else:

        # print("A")
        for i in range(Y_shuffe.shape[0]):
            y = np.zeros((1, class_num), dtype=np.uint8)
            list_target = [c for c in range(class_num)]
            del list_target[int(Y_shuffe[i][0])]

            target_index = random.randint(0, class_num - 2)

            print("A", list_target, Y_shuffe[i], list_target[target_index])
            y[0][list_target[target_index]] = 1
            ys.append(y[0])
            # print(y[0])

        np.save(
            path + "{}_{}_target_labels.npy".format(datasetType, number), np.array(ys)
        )
        print(
            "save the npy file in path :",
            path + "{}_{}_target_labels.npy".format(datasetType, number),
        )

    print(
        "save model is :",
        shuff,
        "\nIsTargeted :",
        IsTargeted,
        "\nsample class number is: ",
        class_num,
        "\nsample total numbers is :{} each type number is : {}".format(
            number, label_one_choice_number
        ),
    )


def load_npy(path_inputs, path_labels):
    origin_nature_samples = np.load(path_inputs)
    origin_labels_samples = np.load(path_labels)

    return origin_nature_samples, origin_labels_samples


#####返回了cifar10的train数据 Xtr,Ytr,cifar10的test数据
Xtr, Ytr, Xte, Yte = load_CIFAR10("../../cifar-10-python/cifar-10-batches-py")

######保存cifar10的test数据的1500个,方式是随机均匀取10类,各150这里,标签是原始的Groundtruth标签,IsTargeted=False######
######如果IsTargeted=True ,则是随机生成和原始样本的GroundTruth不一致的标签,可以用于目标攻击使用,用户也可以自行定义目标标签的生成类别规则#####
save_numpy(
    Xte,
    Yte,
    "../Datasets/CIFAR_cln_data/",
    1500,
    shuff="random_equally",
    datasetType="cifar10",
    IsTargeted=False,
)
#
# cifar100 的调用例子

# numbertest=10000
# Xte100, Yte100=load_CIFAR100('../Datasets/CIFAR10/cifar-100-python','test',numbertest)
# save_numpy( Xte100, Yte100,'../Datasets/cln_data/',300,shuff="random_equally",datasetType="cifar100",IsTargeted=False)


# 加载和显示保存后的数据集的格式
# image_origin_path="../Datasets/cln_data/cifar10_100_origin_inputs.npy"
# label_origin_path="../Datasets/cln_data/cifar10_100_origin_labels.npy"
# origin_nature_samples = np.load(image_origin_path)
# origin_labels_samples = np.load(label_origin_path)
#
# print("sample_shape,label_shape",origin_nature_samples.shape,origin_labels_samples.shape)
