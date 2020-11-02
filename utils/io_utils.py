# -*- coding: utf-8 -*-
import importlib
import numpy as np
from PIL import Image
import os
import sys
import cv2
import random
import json
import string
from .file_utils import inorder_choose_data
import torch


def get_datasets_without_mask(
    imgs_dir, label_dir, height, width, channels, train_test="null"
):
    Nimgs = len(os.listdir(imgs_dir))
    imgs = np.empty((Nimgs, height, width, channels))
    # print(imgs.shape)
    groundTruth = np.empty((Nimgs, height, width))
    for path, subdirs, files in os.walk(
        imgs_dir
    ):  # list all files, directories in the path
        print("images num", len(files))
        for i in range(len(files)):
            # original
            print("image count", i + 1)
            print("original image: " + files[i])
            img = Image.open(os.path.join(imgs_dir, files[i]))
            img = img.convert(mode="RGB")
            # check whether padding is needed
            origin_img = np.asarray(img)
            print("image shape:" + str(origin_img.shape))
            need_padding = False
            imgs[i] = np.asarray(img)
            # corresponding ground truth

            groundTruth_name = "MA_" + files[i].split("_", 1)[-1]
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(os.path.join(label_dir, groundTruth_name))
            origin_manual = np.asarray(g_truth)
            print("manual shape:" + str(origin_manual.shape))
            groundTruth[i] = np.asarray(g_truth)

    #    print ("imgs max: " +str(np.max(imgs)))
    #    print ("imgs min: " +str(np.min(imgs)))
    #    assert(np.max(groundTruth)==255)
    #    assert(np.min(groundTruth)==0)
    print(
        "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    )
    # reshaping for my standard tensors
    # imgs = np.transpose(imgs,(0,3,1,2))
    assert imgs.shape == (Nimgs, height, width, channels)
    # groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    # border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert groundTruth.shape == (Nimgs, height, width)

    return imgs, groundTruth


def get_datasets(
    imgs_dir, label_dir, bordermask_dir, height, width, channels, train_test="null"
):
    Nimgs = len(os.listdir(imgs_dir))
    imgs = np.empty((Nimgs, height, width, channels))
    print(imgs.shape)
    groundTruth = np.empty((Nimgs, height, width))
    border_masks = np.empty((Nimgs, height, width))
    for path, subdirs, files in os.walk(
        imgs_dir
    ):  # list all files, directories in the path
        for i in range(len(files)):
            # original
            print("original image: " + files[i])
            img = Image.open(os.path.join(imgs_dir, files[i]))
            # check whether padding is needed
            origin_img = np.asarray(img)
            print("image shape:" + str(origin_img.shape))
            need_padding = False
            if (origin_img.shape[0] == height) and (origin_img.shape[1] == width):
                imgs[i] = np.asarray(img)
            else:
                need_padding = True
                print("padding image.......")
                origin_img = np.asarray(img)
                target_h, target_w, img_h, img_w = (
                    height,
                    width,
                    origin_img.shape[0],
                    origin_img.shape[1],
                )

                if len(origin_img.shape) == 3:
                    d = origin_img.shape[2]
                    padded_img = np.zeros((target_h, target_w, d))

                elif len(origin_img.shape) == 2:
                    padded_img = np.zeros((target_h, target_w))

                padded_img[
                    (target_h - img_h) // 2 : (target_h - img_h) // 2 + img_h,
                    (target_w - img_w) // 2 : (target_w - img_w) // 2 + img_w,
                    ...,
                ] = origin_img

                # newImage = Image.fromarray(np.uint8(padded_img))
                # newImage.save("/home/rock/devdata/"+files[i])

                imgs[i] = padded_img
            # corresponding ground truth

            groundTruth_name = "manual_" + files[i].split("_")[-1]
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(os.path.join(label_dir, groundTruth_name))
            origin_manual = np.asarray(g_truth)
            print("manual shape:" + str(origin_manual.shape))
            if (origin_manual.shape[0] == height) and (origin_manual.shape[1] == width):
                groundTruth[i] = np.asarray(g_truth)
            else:
                print("padding manual.......")

                target_h, target_w, img_h, img_w = (
                    height,
                    width,
                    origin_manual.shape[0],
                    origin_manual.shape[1],
                )

                if len(origin_manual.shape) == 3:
                    d = origin_manual.shape[2]
                    padded_manual = np.zeros((target_h, target_w, d))

                elif len(origin_manual.shape) == 2:
                    padded_manual = np.zeros((target_h, target_w))

                padded_manual[
                    (target_h - img_h) // 2 : (target_h - img_h) // 2 + img_h,
                    (target_w - img_w) // 2 : (target_w - img_w) // 2 + img_w,
                    ...,
                ] = origin_manual

                groundTruth[i] = padded_manual

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    assert np.max(groundTruth) == 255 and np.max(border_masks) == 255
    assert np.min(groundTruth) == 0 and np.min(border_masks) == 0
    print(
        "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    )
    # reshaping for my standard tensors
    # imgs = np.transpose(imgs,(0,3,1,2))
    assert imgs.shape == (Nimgs, height, width, channels)
    # groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    # border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert groundTruth.shape == (Nimgs, height, width)
    assert border_masks.shape == (Nimgs, height, width)
    return imgs, groundTruth, border_masks


def get_cam_list(self, cam_path, sample_index_list):
    print("get_cam_list")
    cam_list = []
    orig_list = []
    abspath = os.path.abspath(cam_path)
    test_study_list = os.listdir(abspath)
    for study in test_study_list:
        cam_list.append(
            [
                os.path.join(abspath, study, "cam_" + index + ".jpg")
                for index in sample_index_list
            ]
        )
        orig_list.append(
            [
                os.path.join(abspath, study, "orig_" + index + ".jpg")
                for index in sample_index_list
            ]
        )

    return cam_list, orig_list


def get_top_k_list(self, top_k_path):
    print("get_all_test_data_list")
    channels_list = self._config.prepare_data.channels_list
    Nchs = len(channels_list)
    imgs_list = []
    mannual_list = []
    abspath = os.path.abspath(self._config.prepare_data.test_data_dir)
    test_study_list = os.listdir(abspath)
    for study in test_study_list:
        imgs_list.append(
            [os.path.join(abspath, study, ch + ".jpg") for ch in channels_list]
        )
        mannual_list.append([os.path.join(abspath, study, "manual.jpg")])

    # print(imgs_list)
    # print(mannual_list)
    return imgs_list, mannual_list


def result_as_html(base_abspath, x_list, predict_y_list, y_list=None):
    print("result as html")
    # print(x_list)
    # print(predict_y_list)
    # print(y_list)
    assert len(x_list) == len(predict_y_list)
    html_content = (
        "<html><head><title>priction_title</title>"
        '<style type="text/css">.card {float:left;margin: 5px 5px 5px 5px;text-align:center;}'
        "ul {list-style-type:none;}h3.hl {margin-top: 2;margin-bottom: 2;text-align: center;}</style></head><body><ul>"
    )

    for index in range(len(x_list)):
        html_content = html_content + "<li>"
        imgs = x_list[index]
        results = predict_y_list[index]
        assert type(imgs) == type([])
        assert type(results) == type([])
        item_num = len(imgs) + len(results)
        if y_list:
            assert type(y_list[index]) == type([])
            item_num = item_num + len(y_list[index])

        html_img_width = 1800 // item_num - 20
        html_content = (
            html_content
            + '<hr><h3 class="hl">'
            + imgs[0][0 : imgs[0].rfind("/")]
            + "</h3>"
        )
        # imgs = [imgs]
        for i in range(len(imgs)):
            html_content = (
                html_content
                + '<div class="card"><h3>'
                + imgs[i].split("/")[-1]
                + '</h3><img src="'
                + os.path.relpath(imgs[i], base_abspath)
                + '" width='
                + str(html_img_width)
                + " height="
                + str(html_img_width)
                + " /></div>"
            )

        if y_list:
            groundTruth = y_list[index]
            for i in range(len(groundTruth)):
                html_content = (
                    html_content
                    + '<div class="card"><h3>'
                    + groundTruth[i].split("/")[-1]
                    + '</h3><img src="'
                    + os.path.relpath(groundTruth[i], base_abspath)
                    + '" width='
                    + str(html_img_width)
                    + " height="
                    + str(html_img_width)
                    + " /></div>"
                )

        for i in range(len(results)):
            html_content = (
                html_content
                + '<div class="card"><h3>'
                + results[i].split("/")[-1]
                + '</h3><img src="'
                + os.path.relpath(results[i], base_abspath)
                + '" width='
                + str(html_img_width)
                + " height="
                + str(html_img_width)
                + " /></div>"
            )

        html_content = html_content + "</li>"

    html_content = html_content + "</ul></body></html>"

    return html_content


def get_resized_img_from_dir(
    MSI_filename, MSI_image_name, original_row, original_col, resizeratio
):

    resize_row = int(original_row / resizeratio)
    resize_col = int(original_col / resizeratio)

    img_data = np.ndarray((resize_row, resize_col, len(MSI_image_name)), dtype=np.uint8)
    print("MSI_filename:", MSI_filename)
    if len(MSI_image_name) > 1:
        for i in range(len(MSI_image_name)):
            MSI_image = cv2.imread(MSI_filename + "/" + MSI_image_name[i] + ".jpg")
            if resizeratio > 1 or resizeratio < 1:
                MSI_image = cv2.resize(MSI_image, (resize_row, resize_col))
            img_data[:, :, i] = MSI_image[:, :, 0]
    else:
        MSI_image = cv2.imread(MSI_filename + "/" + MSI_image_name[0] + ".jpg")
        if resizeratio > 1 or resizeratio < 1:
            MSI_image = cv2.resize(MSI_image, (resize_row, resize_col))
        img_data = MSI_image

    print("img_data:", img_data.shape)
    return img_data


def subimage(image, center, theta, width, height):
    theta *= np.pi / 180  # convert to rad

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(
        image,
        mapping,
        (width, height),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def CreateSave_dir(MSI_image_save_file, PosOrNeg, patch_size, scale_ratio):

    patch_name0 = PosOrNeg
    save_dir0 = os.path.join(MSI_image_save_file, patch_name0)
    if os.path.exists(save_dir0) == False:
        os.makedirs(save_dir0)

    patch_name1 = str(patch_size)
    save_dir1 = os.path.join(MSI_image_save_file, patch_name0, patch_name1)
    if os.path.exists(save_dir1) == False:
        os.makedirs(save_dir1)

    save_dir2 = os.path.join(MSI_image_save_file, patch_name0, patch_name1, "images")
    if os.path.exists(save_dir2) == False:
        os.makedirs(save_dir2)

    save_dir3 = os.path.join(MSI_image_save_file, patch_name0, patch_name1, "masks")
    if os.path.exists(save_dir3) == False:
        os.makedirs(save_dir3)

    if scale_ratio > 0:
        patch_name2 = str(int(patch_size * scale_ratio))

        save_dir1 = os.path.join(MSI_image_save_file, patch_name0, patch_name2)
        if os.path.exists(save_dir1) == False:
            os.makedirs(save_dir1)

        save_dir2 = os.path.join(
            MSI_image_save_file, patch_name0, patch_name2, "images"
        )
        if os.path.exists(save_dir2) == False:
            os.makedirs(save_dir2)

        save_dir3 = os.path.join(MSI_image_save_file, patch_name0, patch_name2, "masks")
        if os.path.exists(save_dir3) == False:
            os.makedirs(save_dir3)


def SaveImageName(File, PosOrNeg, patch_size, scale_ratio):
    save_dir_image2 = ""
    patch_name1 = str(patch_size)
    print("PosOrNeg:", PosOrNeg)
    save_dir_image = os.path.join(File, PosOrNeg, patch_name1, "images")

    if scale_ratio > 0:
        patch_name2 = str(int(patch_size * scale_ratio))
        save_dir_image2 = os.path.join(File, PosOrNeg, patch_name2, "images")

    return save_dir_image, save_dir_image2


def SaveMaskName(File, PosOrNeg, patch_size, scale_ratio):
    save_dir_mask2 = ""
    patch_name1 = str(patch_size)
    save_dir_mask = os.path.join(File, PosOrNeg, patch_name1, "masks")

    if scale_ratio > 0:
        patch_name2 = str(int(patch_size * scale_ratio))
        save_dir_mask2 = os.path.join(File, PosOrNeg, patch_name2, "masks")

    return save_dir_mask, save_dir_mask2


def SaveWithJson(
    DIR, save_type, table_name="", model_name="", evaluation_name="", row="", value=0
):
    # 检测结果文件是否存在
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(DIR + "/result.txt"):
        with open(DIR + "/result.txt", "w") as file:
            json.dump({}, file)
    file = open(DIR + "/result.txt", "r")
    js = file.read()
    file.close()
    dic = json.loads(js)
    if not save_type in dic:
        dic[save_type] = {}
    # 存储内容为基础评测
    if save_type == "table_list":
        if not table_name in dic[save_type]:
            dic[save_type][table_name] = {}
            model_name = model_name.split(".")[-1]
            print(model_name)
            dic[save_type][table_name]["TITLE"] = [model_name, "CLEAN", table_name, []]
            dic[save_type][table_name][row] = [row, 0]
        if not row in dic[save_type][table_name]:
            dic[save_type][table_name][row] = [row, 0]
        if evaluation_name == "CLEAN ACC":
            dic[save_type][table_name][row][1] = value
        else:
            dic[save_type][table_name]["TITLE"][3].append(evaluation_name)
            if type(value) == list:
                value = value[0]
            dic[save_type][table_name][row].append((value))

    # 存储内容为热力图分析
    elif save_type == "cam":
        pass

    # 存储内容为mCE分析
    elif save_type == "mCE":
        pass

    # 存储内容为EENI分析
    elif save_type == "EENI":
        pass

    with open(DIR + "/result.txt", "w") as file:
        file.write(json.dumps(dic, ensure_ascii=False))


# SaveWithJson_Result(args.save_path, "table_list", attName[0], f, args.evaluation_method, "evaluation result", rst)
def SaveWithJson_Result(
    DIR, save_type, attName="", Attack_file_name="", evaluation_name="", value=0
):
    # 检测结果文件是否存在
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(DIR + "/result.txt"):
        with open(DIR + "/result.txt", "w") as file:
            json.dump({}, file)
    file = open(DIR + "/result.txt", "r")
    js = file.read()
    file.close()
    dic = json.loads(js)
    if not save_type in dic:
        dic[save_type] = {}
    # 存储内容为基础评测
    att_param_name = Attack_file_name.split(".")[0]
    print(att_param_name)
    if save_type == "table_list":
        if not attName in dic[save_type]:
            dic[save_type][attName] = {}
            dic[save_type][attName][att_param_name] = {}
            dic[save_type][attName][att_param_name] = [[], []]
        if not att_param_name in dic[save_type][attName]:
            dic[save_type][attName][att_param_name] = [[evaluation_name], []]
            dic[save_type][attName][att_param_name][1] = [value]
        else:
            dic[save_type][attName][att_param_name][0].append(evaluation_name)
            if type(value) == list:
                value = value[0]
            dic[save_type][attName][att_param_name][1].append((value))

    with open(DIR + "/result.txt", "w") as file:
        file.write(json.dumps(dic, ensure_ascii=False))


def update_current_status(DIR, attack_method, value):
    #  检查文件是否存在
    if not os.path.exists(DIR + "/temp.txt"):
        return
    with open(DIR + "/temp.txt", "r") as file:
        js = file.read()
    dic = json.loads(js)
    if attack_method in dic:
        dic[attack_method] = value
    with open(DIR + "/temp.txt", "w") as file:
        file.write(json.dumps(dic))


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False


def get_label_lines(path_label):
    sample_names, labels = inorder_choose_data(path_label, 1, division=" ")
    return sample_names, labels, len(labels)


def center_Crop(Image, ImageScale_size, crop_size):
    imgcv = cv2.resize(Image, (ImageScale_size[0], ImageScale_size[1]))
    if ImageScale_size == crop_size:
        return imgcv
    center_x = imgcv.shape[0] // 2
    center_y = imgcv.shape[1] // 2
    cropImg = imgcv[
        center_x - crop_size[0] // 2 : center_x + crop_size[0] // 2,
        center_y - crop_size[1] // 2 : center_y + crop_size[1] // 2,
    ]
    return cropImg


def get_image_from_path(ImagePath, index, ImageScale_size, crop_size):
    pathnames = os.listdir(ImagePath)
    pathname = pathnames[index]
    image = cv2.imread(ImagePath + "/" + pathname)
    imgcv = center_Crop(image, ImageScale_size, crop_size)
    image = np.ascontiguousarray(np.transpose(imgcv, (2, 0, 1)))
    image = np.float32(image) / 255
    return image, imgcv


def convertlist_to_numpy(list_inputs):
    outputs_numpy = []
    for i in range(len(list_inputs)):
        output_numpy = list_inputs[i].cpu().numpy()[np.newaxis, :]
        outputs_numpy.extend(output_numpy)
    return np.array(outputs_numpy)


def save_json(path, tensor_data):
    with open(path, "w") as jsonFile:
        json.dump({"data": tensor_data.tolist()}, jsonFile)


def read_json(path, dict_name):
    with open(path, "r") as load_f:
        load_dict = json.load(load_f)
    # print(np.array(load_dict[dict_name]).shape)
    return np.array(load_dict[dict_name])


def load_json(path):
    with open(path, "r") as load_f:
        load_dict = json.load(load_f)
    # print(np.array(load_dict[dict_name]).shape)
    return load_dict


def gen_attack_adv_save_path(save_base_path, args_Attack_param):
    path_origins = str(args_Attack_param).split(".")[0].split("/")
    mkdir(save_base_path)
    new_base_path = save_base_path + path_origins[0]
    for i in range(len(path_origins) - 1):
        new_base_path = new_base_path + "_" + path_origins[i + 1]
    return new_base_path


def analyze_json(jsons):
    """
    解析传进来的jsons,将jsons解析成key-value并输出
    :param jsons: 需要解析的json字符串
    :return:
    """
    key_value = ""
    # isinstance函数是Python的内部函数，他的作用是判断jsons这个参数是否为dict类型
    # 如果是的话返回True，否则返回False
    if isinstance(jsons, dict):
        for key in jsons.keys():
            key_value = jsons.get(key)
            if isinstance(key_value, dict):
                analyze_json(key_value)
            elif isinstance(key_value, list):
                for json_array in key_value:
                    analyze_json(json_array)
            else:
                print(str(key) + " = " + str(key_value))
    elif isinstance(jsons, list):
        for json_array in jsons:
            analyze_json(json_array)


def output_value(jsons, key):
    """
    通过参数key，在jsons中进行匹配并输出该key对应的value
    :param jsons: 需要解析的json串
    :param key: 需要查找的key
    :return:
    """
    key_value = ""
    if isinstance(jsons, dict):
        for json_result in jsons.values():
            if key in jsons.keys():
                key_value = jsons.get(key)
            else:
                output_value(json_result, key)
    elif isinstance(jsons, list):
        for json_array in jsons:
            output_value(json_array, key)
    if key_value != "":
        # print(str(key) + " = " + str(key_value))
        return key_value


def dict_list_to_np(dict_list):
    outputs = []
    for single_output in dict_list:
        outputs.append(single_output)
    return np.array(outputs, dtype=np.float32)


def configurate_Device(seed, gpu_counts, gpu_indexs):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device_counts = torch.cuda.device_count()
        if device_counts < int(gpu_counts):
            print("Can't set the gpu number larger than the available numbers")
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                return device
        else:
            gpu_indexs = str(gpu_indexs).split(",")
            np_gpu_indexs = np.array(gpu_indexs).astype(int)
            # gpu的设备号满足个数要求，但是要看看是否满足别的
            if min(np_gpu_indexs) >= 0 and max(np_gpu_indexs) < int(gpu_counts):
                rand_index = random.randint(0, int(gpu_counts))
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indexs[rand_index])
                # for i in range(np_gpu_indexs.shape[0]):
                #     devicename=torch.cuda.get_device_name(i)
            else:
                # 名称设置的过多，或者没有和实际用的设备同名
                # 设备号的设备名称为准，如果和设备的类型一致，就用用户的
                # Set the random seed manually for reproducibility.
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
    return device
