import importlib
import numpy as np
from PIL import Image
import h5py
import os
import sys
import cv2
import random


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def get_datasets_without_mask(
    imgs_dir, label_dir, height, width, channels, train_test="null"
):
    Nimgs = len(os.listdir(imgs_dir))
    imgs = np.empty((Nimgs, height, width, channels))
    print(imgs.shape)
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


# def GenTrainImages_positive(MSI_image,Mask_image, MSI_image_save_file, PosOrNeg, case_num, patch_size, scale_ratio = -1):
#
#     save_dir_image, save_dir_image2 = SaveImageName(MSI_image_save_file, PosOrNeg,patch_size, scale_ratio)
#     save_dir_mask, save_dir_mask2 = SaveMaskName(MSI_image_save_file, PosOrNeg,patch_size, scale_ratio)
#
#     BLACK = [0, 0, 0]
#     Mask_image_padding = cv2.copyMakeBorder(Mask_image, int(patch_size*1.5), int(patch_size*1.5), int(patch_size*1.5), int(patch_size*1.5), cv2.BORDER_CONSTANT, value=BLACK)
#     MSI_image_padding = cv2.copyMakeBorder(MSI_image, int(patch_size*1.5), int(patch_size*1.5), int(patch_size*1.5), int(patch_size*1.5),
#                                             cv2.BORDER_CONSTANT, value=BLACK)
#
#     gray = cv2.cvtColor(Mask_image_padding, cv2.COLOR_BGR2GRAY)
#     _, binary_origin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#     binary, contours, hierarchy = cv2.findContours(binary_origin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Reference: https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
#     crop_num = 0
#     print('len(contours):', len(contours))
#     for i in range(0, len(contours)):
#
#         ect = cv2.minAreaRect(contours[i])
#         center = [int(ect[0][0]), int(ect[0][1])]
#         print('ect:', center)
#
#         ############################ crop centered image
#         crop_mask = Mask_image_padding[(center[1]-int(patch_size/2)):(center[1]+int(patch_size/2)), (center[0]-int(patch_size/2)):(center[0]+int(patch_size/2))]
#         crop_mask_name = save_dir_mask + '/' + str(case_num) + '_C_' + str(i) + '_' + str(crop_num) + '.jpg'
#         cv2.imwrite(crop_mask_name, crop_mask)
#         crop_MSI_image = MSI_image_padding[(center[1] - int(patch_size / 2)):(center[1] + int(patch_size / 2)),
#                     (center[0] - int(patch_size / 2)):(center[0] + int(patch_size / 2))]
#         crop_MSI_name = save_dir_image + '/' + str(case_num) + '_C_' + str(i) + '_' + str(crop_num) + '.jpg'
#         cv2.imwrite(crop_MSI_name, crop_MSI_image)
#
#         ############################ crop centered image and rotate
#         if aug_rotation == 'yes':
#             rotate_theta = float(random.sample(range(rotation_theta, 360, rotation_theta), 1)[0])
#             crop_mask_R = subimage(Mask_image_padding, center, rotate_theta, patch_size, patch_size)
#             crop_mask_name = save_dir_mask + '/' + str(case_num) + '_CR_' + str(i) + '_' + str(crop_num) + '.jpg'
#             cv2.imwrite(crop_mask_name, crop_mask_R)
#
#             crop_MSI_image_R = subimage(MSI_image_padding, center, rotate_theta, patch_size, patch_size)
#             crop_MSI_name = save_dir_image + '/' + str(case_num) + '_CR_' + str(i) + '_' + str(crop_num) + '.jpg'
#             cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
#
#
#             # crop another scale centered image
#         if scale_ratio > -1:
#             crop_mask = Mask_image_padding[(center[1] - int(patch_size*scale_ratio/ 2)):(center[1] + int(patch_size*scale_ratio/2)),
#                         (center[0] - int(patch_size*scale_ratio / 2)):(center[0] + int(patch_size*scale_ratio / 2))]
#             crop_mask_name = save_dir_mask2 + '/' + str(case_num) + '_C_' + str(i) + '_' + str(crop_num) + '.jpg'
#             cv2.imwrite(crop_mask_name, crop_mask)
#             crop_MSI_image = MSI_image_padding[(center[1] - int(patch_size*scale_ratio / 2)):(center[1] + int(patch_size*scale_ratio / 2)),
#                              (center[0] - int(patch_size*scale_ratio / 2)):(center[0] + int(patch_size*scale_ratio / 2))]
#             crop_MSI_name = save_dir_image2 + '/' + str(case_num) + '_C_' + str(i) + '_' + str(crop_num) + '.jpg'
#             cv2.imwrite(crop_MSI_name, crop_MSI_image)
#
#             ############################ crop another scale centered image and rotate
#             if aug_rotation == 'yes':
#                 crop_mask_R = subimage(Mask_image_padding, center, rotate_theta, int(patch_size * scale_ratio),
#                                        int(patch_size * scale_ratio))
#                 crop_mask_name = save_dir_mask2 + '/' + str(case_num) + '_CR_' + str(i) + '_' + str(crop_num) + '.jpg'
#                 cv2.imwrite(crop_mask_name, crop_mask_R)
#
#                 crop_MSI_image_R = subimage(MSI_image_padding, center, rotate_theta, int(patch_size * scale_ratio),
#                                             int(patch_size * scale_ratio))
#                 crop_MSI_name = save_dir_image2 + '/' + str(case_num) + '_CR_' + str(i) + '_' + str(crop_num) + '.jpg'
#                 cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
#
#
#
#         ############################ crop translation images
#         if aug_translation == 'yes':
#
#             translation_len = random.sample(range(trans_lengthelement, int(patch_size/2-trans_lengthelement), trans_lengthelement), aug_translation_num)
#             translation_theta = random.sample(range(trans_theta, 360, trans_theta), aug_translation_num)
#
#             print('translation_len:', translation_len)
#             print('rotate_theta:', translation_theta)
#             ############################ crop translation-centered images
#             for tri in range(aug_translation_num):
#                 print('center:', center)
#                 print('tri:',translation_len[tri])
#                 print('tri:', translation_theta[tri])
#
#                 coord_trans_x = center[0] + np.cos((translation_theta[tri]* np.pi / 180))*translation_len[tri]
#                 coord_trans_y  = center[1] + np.sin((translation_theta[tri]* np.pi / 180)) * translation_len[tri]
#
#                 center_trans = [int(coord_trans_x), int(coord_trans_y)]
#
#                 print('center_trans:',center_trans)
#
#                 crop_mask = Mask_image_padding[(center_trans[1] - int(patch_size / 2)):(center_trans[1] + int(patch_size / 2)),
#                             (center_trans[0] - int(patch_size / 2)):(center_trans[0] + int(patch_size / 2))]
#                 crop_mask_name = save_dir_mask + '/' + str(case_num) + '_CT_' + str(i) + '_' + str(crop_num) + '_' + str(tri) + '.jpg'
#                 cv2.imwrite(crop_mask_name, crop_mask)
#                 crop_MSI_image = MSI_image_padding[(center_trans[1] - int(patch_size / 2)):(center_trans[1] + int(patch_size / 2)),
#                                  (center_trans[0] - int(patch_size / 2)):(center_trans[0] + int(patch_size / 2))]
#                 crop_MSI_name = save_dir_image + '/' + str(case_num) + '_CT_' + str(i) + '_' + str(crop_num) + '_' + str(tri) +'.jpg'
#                 cv2.imwrite(crop_MSI_name, crop_MSI_image)
#
#                 ############################ crop centered image and rotate
#                 if aug_rotation == 'yes':
#                     rotate_theta = float(random.sample(range(rotation_theta, 360, rotation_theta), 1)[0])
#                     crop_mask_R = subimage(Mask_image_padding, center_trans, rotate_theta, patch_size, patch_size)
#                     crop_mask_name = save_dir_mask + '/' + str(case_num) + '_CTR_' + str(i) + '_' + str(
#                         crop_num) + '_' + str(tri) + '.jpg'
#                     cv2.imwrite(crop_mask_name, crop_mask_R)
#
#                     crop_MSI_image_R = subimage(MSI_image_padding, center_trans, rotate_theta, patch_size, patch_size)
#                     crop_MSI_name = save_dir_image + '/' + str(case_num) + '_CTR_' + str(i) + '_' + str(
#                         crop_num) + '_' + str(tri) + '.jpg'
#                     cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
#
#                 ################### crop another scale centered image
#                 if scale_ratio > -1:
#                     crop_mask = Mask_image_padding[(center_trans[1] - int(patch_size * scale_ratio / 2)):(
#                         center_trans[1] + int(patch_size * scale_ratio / 2)),
#                                 (center_trans[0] - int(patch_size * scale_ratio / 2)):(
#                                     center_trans[0] + int(patch_size * scale_ratio / 2))]
#                     crop_mask_name = save_dir_mask2 + '/' + str(case_num) + '_CT_' + str(i) + '_' + str(
#                         crop_num) + '_' + str(tri) + '.jpg'
#                     cv2.imwrite(crop_mask_name, crop_mask)
#                     crop_MSI_image = MSI_image_padding[(center_trans[1] - int(patch_size * scale_ratio / 2)):(
#                         center_trans[1] + int(patch_size * scale_ratio / 2)),
#                                      (center_trans[0] - int(patch_size * scale_ratio / 2)):(
#                                          center_trans[0] + int(patch_size * scale_ratio / 2))]
#                     crop_MSI_name = save_dir_image2 + '/' + str(case_num) + '_CT_' + str(i) + '_' + str(
#                         crop_num) + '_' + str(tri) + '.jpg'
#                     cv2.imwrite(crop_MSI_name, crop_MSI_image)
#
#                     ############################ crop another scale centered image and rotate
#                     if aug_rotation == 'yes':
#                         crop_mask_R = subimage(Mask_image_padding, center_trans, rotate_theta, int(patch_size * scale_ratio),
#                                                int(patch_size * scale_ratio))
#                         crop_mask_name = save_dir_mask2 + '/' + str(case_num) + '_CTR_' + str(i) + '_' + str(
#                             crop_num) + '_' + str(tri) + '.jpg'
#                         cv2.imwrite(crop_mask_name, crop_mask_R)
#
#                         crop_MSI_image_R = subimage(MSI_image_padding, center_trans, rotate_theta,
#                                                     int(patch_size * scale_ratio),
#                                                     int(patch_size * scale_ratio))
#                         crop_MSI_name = save_dir_image2 + '/' + str(case_num) + '_CTR_' + str(i) + '_' + str(
#                             crop_num) + '_' + str(tri) + '.jpg'
#                         cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
#                 print('Iteration:')
#         crop_num = crop_num + 1
#
# def GenTrainImages_negative(MSI_image, Mask_image, MSI_image_save_file, PosOrNeg, case_num, patch_size, step = 10, scale_ratio=-1):
#
#     save_dir_image, save_dir_image2 = SaveImageName(MSI_image_save_file, PosOrNeg, patch_size, scale_ratio)
#     save_dir_mask, save_dir_mask2 = SaveMaskName(MSI_image_save_file, PosOrNeg, patch_size, scale_ratio)
#
#     BLACK = [0, 0, 0]
#     Mask_image_padding = cv2.copyMakeBorder(Mask_image, int(patch_size * 1.5), int(patch_size * 1.5),
#                                             int(patch_size * 1.5), int(patch_size * 1.5), cv2.BORDER_CONSTANT,
#                                             value=BLACK)
#     MSI_image_padding = cv2.copyMakeBorder(MSI_image, int(patch_size * 1.5), int(patch_size * 1.5),
#                                            int(patch_size * 1.5), int(patch_size * 1.5),
#                                            cv2.BORDER_CONSTANT, value=BLACK)
#
#     size_1 = int(patch_size * 1.5)
#     size_2 = int((patch_size * scale_ratio-patch_size)/2)
#     ##########################
#     height = Mask_image.shape[0]
#     width = Mask_image.shape[1]
#
#     w = patch_size
#     h = patch_size
#
#     W = int(patch_size* scale_ratio)
#     H = int(patch_size * scale_ratio)
#
#     for row in range(0,height-patch_size,step):
#         for col in range(0,width-patch_size,step):
#             x = col
#             y = row
#
#             if (row + patch_size) > height:
#                 y = height - patch_size
#             if (col + patch_size) > width:
#                 x = width - patch_size
#
#
#             crop_mask = Mask_image[y:y + h, x:x + w]  # 先用y确定高，再用x确定宽
#             print('crop_mask:', crop_mask.shape)
#
#             flag = 0
#             for sub_row in range(0,patch_size):
#                 for sub_col in range(0,patch_size):
#                     if crop_mask[sub_row,sub_col,0] >=200:
#                         flag = 1
#                         break
#
#             if flag == 0:
#                 crop_mask = Mask_image_padding[(y + size_1):(y + h + size_1),
#                             (x + size_1):(x + w + size_1)]  # 先用y确定高，再用x确定宽
#                 crop_mask_name = save_dir_mask + '/' + str(case_num) + '_Neg_' + str(row) + '_' + str(col) + '.jpg'
#                 cv2.imwrite(crop_mask_name, crop_mask)
#
#                 crop_MSI_image = MSI_image_padding[(y + size_1):(y + h + size_1),
#                                  (x + size_1):(x + w + size_1)]  # 先用y确定高，再用x确定宽
#                 crop_MSI_image_name = save_dir_image + '/' + str(case_num) + '_Neg_' + str(row) + '_' + str(
#                     col) + '.jpg'
#                 cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
#
#                 if scale_ratio > -1:
#                     crop_mask = Mask_image_padding[(y - size_2 + size_1):(y - size_2 + H + size_1),
#                                 (x - size_2 + size_1):(x - size_2 + W + size_1)]  # 先用y确定高，再用x确定宽
#                     crop_mask_name = save_dir_mask2 + '/' + str(case_num) + '_Neg_' + str(row) + '_' + str(col) + '.jpg'
#                     cv2.imwrite(crop_mask_name, crop_mask)
#
#                     crop_MSI_image = MSI_image_padding[(y- size_2 + size_1):(y- size_2 + H + size_1),
#                                      (x- size_2 + size_1):(x- size_2 + W + size_1)]  # 先用y确定高，再用x确定宽
#                     crop_MSI_image_name = save_dir_image2 + '/' + str(case_num) + '_Neg_' + str(row) + '_' + str(
#                         col) + '.jpg'
#                     cv2.imwrite(crop_MSI_image_name, crop_MSI_image)


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
