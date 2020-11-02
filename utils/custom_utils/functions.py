from PIL import Image
import cv2
import os
import shutil
import pdb
import random
import numpy as np
from scipy import misc

aug_rotation = "yes"
rotation_theta = 20
aug_rotation_num = 10
aug_noise = "no"

aug_translation = "yes"
aug_translation_num = 6
trans_theta = 45
trans_lengthelement = 2


def MultiChannelData(
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


def GenTrainImages_positive(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    scale_ratio=-1,
):

    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []

    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )

    gray = cv2.cvtColor(Mask_image_padding, cv2.COLOR_BGR2GRAY)
    _, binary_origin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(
        binary_origin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Reference: https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
    crop_num = 0
    print("len(contours):", len(contours))
    for i in range(0, len(contours)):

        ect = cv2.minAreaRect(contours[i])
        center = [int(ect[0][0]), int(ect[0][1])]
        print("ect:", center)

        ############################ crop centered image
        crop_mask = Mask_image_padding[
            (center[1] - int(patch_size / 2)) : (center[1] + int(patch_size / 2)),
            (center[0] - int(patch_size / 2)) : (center[0] + int(patch_size / 2)),
            0,
        ]
        crop_mask_name = (
            save_dir_mask
            + "/"
            + str(case_num)
            + "_C_"
            + str(i)
            + "_"
            + str(crop_num)
            + ".jpg"
        )
        cv2.imwrite(crop_mask_name, crop_mask)
        mask_list.append(crop_mask)
        crop_MSI_image = MSI_image_padding[
            (center[1] - int(patch_size / 2)) : (center[1] + int(patch_size / 2)),
            (center[0] - int(patch_size / 2)) : (center[0] + int(patch_size / 2)),
        ]
        crop_MSI_name = (
            save_dir_image
            + "/"
            + str(case_num)
            + "_C_"
            + str(i)
            + "_"
            + str(crop_num)
            + ".jpg"
        )
        cv2.imwrite(crop_MSI_name, crop_MSI_image)
        img_list.append(crop_MSI_image)
        ############################ crop centered image and rotate
        if aug_rotation == "yes":
            rotate_theta_random = random.sample(
                range(rotation_theta, 360, rotation_theta), aug_rotation_num
            )
            for roi in range(aug_rotation_num):
                rotate_theta = float(rotate_theta_random[roi])
                crop_mask_R = subimage(
                    Mask_image_padding, center, rotate_theta, patch_size, patch_size
                )
                crop_mask_R = crop_mask_R[:, :, 0]
                crop_mask_name = (
                    save_dir_mask
                    + "/"
                    + str(case_num)
                    + "_CR_"
                    + str(i)
                    + "_"
                    + str(roi)
                    + "_"
                    + str(crop_num)
                    + ".jpg"
                )
                cv2.imwrite(crop_mask_name, crop_mask_R)
                mask_list.append(crop_mask_R)
                crop_MSI_image_R = subimage(
                    MSI_image_padding, center, rotate_theta, patch_size, patch_size
                )
                crop_MSI_name = (
                    save_dir_image
                    + "/"
                    + str(case_num)
                    + "_CR_"
                    + str(i)
                    + "_"
                    + str(roi)
                    + "_"
                    + str(crop_num)
                    + ".jpg"
                )
                cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                img_list.append(crop_MSI_image_R)

            # crop another scale centered image
        if scale_ratio > -1:
            crop_mask = Mask_image_padding[
                (center[1] - int(patch_size * scale_ratio / 2)) : (
                    center[1] + int(patch_size * scale_ratio / 2)
                ),
                (center[0] - int(patch_size * scale_ratio / 2)) : (
                    center[0] + int(patch_size * scale_ratio / 2)
                ),
                0,
            ]
            crop_mask_name = (
                save_dir_mask2
                + "/"
                + str(case_num)
                + "_C_"
                + str(i)
                + "_"
                + str(crop_num)
                + ".jpg"
            )
            cv2.imwrite(crop_mask_name, crop_mask)
            sub_mask_list.append(crop_mask)
            crop_MSI_image = MSI_image_padding[
                (center[1] - int(patch_size * scale_ratio / 2)) : (
                    center[1] + int(patch_size * scale_ratio / 2)
                ),
                (center[0] - int(patch_size * scale_ratio / 2)) : (
                    center[0] + int(patch_size * scale_ratio / 2)
                ),
            ]
            crop_MSI_name = (
                save_dir_image2
                + "/"
                + str(case_num)
                + "_C_"
                + str(i)
                + "_"
                + str(crop_num)
                + ".jpg"
            )
            cv2.imwrite(crop_MSI_name, crop_MSI_image)
            sub_img_list.append(crop_MSI_image)
            ############################ crop another scale centered image and rotate
            if aug_rotation == "yes":
                # rotate_theta_random = random.sample(range(rotation_theta, 360, rotation_theta), aug_rotation_num)
                for roi in range(aug_rotation_num):
                    crop_mask_R = subimage(
                        Mask_image_padding,
                        center,
                        rotate_theta,
                        int(patch_size * scale_ratio),
                        int(patch_size * scale_ratio),
                    )
                    crop_mask_R = crop_mask_R[:, :, 0]
                    crop_mask_name = (
                        save_dir_mask2
                        + "/"
                        + str(case_num)
                        + "_CR_"
                        + str(i)
                        + "_"
                        + str(roi)
                        + "_"
                        + str(crop_num)
                        + ".jpg"
                    )
                    cv2.imwrite(crop_mask_name, crop_mask_R)
                    sub_mask_list.append(crop_mask_R)

                    crop_MSI_image_R = subimage(
                        MSI_image_padding,
                        center,
                        rotate_theta,
                        int(patch_size * scale_ratio),
                        int(patch_size * scale_ratio),
                    )
                    crop_MSI_name = (
                        save_dir_image2
                        + "/"
                        + str(case_num)
                        + "_CR_"
                        + str(i)
                        + "_"
                        + str(roi)
                        + "_"
                        + str(crop_num)
                        + ".jpg"
                    )
                    cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                    sub_img_list.append(crop_MSI_image_R)

        ############################ crop translation images
        if aug_translation == "yes":

            translation_len = random.sample(
                range(
                    trans_lengthelement,
                    int(patch_size / 2 - trans_lengthelement),
                    trans_lengthelement,
                ),
                aug_translation_num,
            )
            translation_theta = random.sample(
                range(trans_theta, 360, trans_theta), aug_translation_num
            )

            print("translation_len:", translation_len)
            print("rotate_theta:", translation_theta)
            ############################ crop translation-centered images
            for tri in range(aug_translation_num):
                print("center:", center)
                print("tri:", translation_len[tri])
                print("tri:", translation_theta[tri])

                coord_trans_x = (
                    center[0]
                    + np.cos((translation_theta[tri] * np.pi / 180))
                    * translation_len[tri]
                )
                coord_trans_y = (
                    center[1]
                    + np.sin((translation_theta[tri] * np.pi / 180))
                    * translation_len[tri]
                )

                center_trans = [int(coord_trans_x), int(coord_trans_y)]

                print("center_trans:", center_trans)

                crop_mask = Mask_image_padding[
                    (center_trans[1] - int(patch_size / 2)) : (
                        center_trans[1] + int(patch_size / 2)
                    ),
                    (center_trans[0] - int(patch_size / 2)) : (
                        center_trans[0] + int(patch_size / 2)
                    ),
                    0,
                ]
                crop_mask_name = (
                    save_dir_mask
                    + "/"
                    + str(case_num)
                    + "_CT_"
                    + str(i)
                    + "_"
                    + str(crop_num)
                    + "_"
                    + str(tri)
                    + ".jpg"
                )
                cv2.imwrite(crop_mask_name, crop_mask)
                mask_list.append(crop_mask)
                crop_MSI_image = MSI_image_padding[
                    (center_trans[1] - int(patch_size / 2)) : (
                        center_trans[1] + int(patch_size / 2)
                    ),
                    (center_trans[0] - int(patch_size / 2)) : (
                        center_trans[0] + int(patch_size / 2)
                    ),
                ]
                crop_MSI_name = (
                    save_dir_image
                    + "/"
                    + str(case_num)
                    + "_CT_"
                    + str(i)
                    + "_"
                    + str(crop_num)
                    + "_"
                    + str(tri)
                    + ".jpg"
                )
                cv2.imwrite(crop_MSI_name, crop_MSI_image)
                img_list.append(crop_MSI_image)
                ############################ crop centered image and rotate
                if aug_rotation == "yes":
                    rotate_theta_random = random.sample(
                        range(rotation_theta, 360, rotation_theta), aug_rotation_num
                    )
                    for roi in range(aug_rotation_num):
                        rotate_theta = float(rotate_theta_random[roi])
                        # rotate_theta = float(random.sample(range(rotation_theta, 360, rotation_theta), 1)[0])
                        crop_mask_R = subimage(
                            Mask_image_padding,
                            center_trans,
                            rotate_theta,
                            patch_size,
                            patch_size,
                        )
                        crop_mask_R = crop_mask_R[:, :, 0]
                        crop_mask_name = (
                            save_dir_mask
                            + "/"
                            + str(case_num)
                            + "_CTR_"
                            + str(i)
                            + "_"
                            + str(roi)
                            + "_"
                            + str(crop_num)
                            + "_"
                            + str(tri)
                            + ".jpg"
                        )
                        cv2.imwrite(crop_mask_name, crop_mask_R)
                        mask_list.append(crop_mask_R)

                        crop_MSI_image_R = subimage(
                            MSI_image_padding,
                            center_trans,
                            rotate_theta,
                            patch_size,
                            patch_size,
                        )
                        crop_MSI_name = (
                            save_dir_image
                            + "/"
                            + str(case_num)
                            + "_CTR_"
                            + str(i)
                            + "_"
                            + str(roi)
                            + "_"
                            + str(crop_num)
                            + "_"
                            + str(tri)
                            + ".jpg"
                        )
                        cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                        img_list.append(crop_MSI_image_R)

                ################### crop another scale centered image
                if scale_ratio > -1:
                    crop_mask = Mask_image_padding[
                        (center_trans[1] - int(patch_size * scale_ratio / 2)) : (
                            center_trans[1] + int(patch_size * scale_ratio / 2)
                        ),
                        (center_trans[0] - int(patch_size * scale_ratio / 2)) : (
                            center_trans[0] + int(patch_size * scale_ratio / 2)
                        ),
                        0,
                    ]
                    crop_mask_name = (
                        save_dir_mask2
                        + "/"
                        + str(case_num)
                        + "_CT_"
                        + str(i)
                        + "_"
                        + str(crop_num)
                        + "_"
                        + str(tri)
                        + ".jpg"
                    )
                    cv2.imwrite(crop_mask_name, crop_mask)
                    sub_mask_list.append(crop_mask)
                    crop_MSI_image = MSI_image_padding[
                        (center_trans[1] - int(patch_size * scale_ratio / 2)) : (
                            center_trans[1] + int(patch_size * scale_ratio / 2)
                        ),
                        (center_trans[0] - int(patch_size * scale_ratio / 2)) : (
                            center_trans[0] + int(patch_size * scale_ratio / 2)
                        ),
                    ]
                    crop_MSI_name = (
                        save_dir_image2
                        + "/"
                        + str(case_num)
                        + "_CT_"
                        + str(i)
                        + "_"
                        + str(crop_num)
                        + "_"
                        + str(tri)
                        + ".jpg"
                    )
                    cv2.imwrite(crop_MSI_name, crop_MSI_image)
                    sub_img_list.append(crop_MSI_image)

                    ############################ crop another scale centered image and rotate
                    if aug_rotation == "yes":
                        # rotate_theta_random = random.sample(range(rotation_theta, 360, rotation_theta),
                        #                                     aug_rotation_num)
                        for roi in range(aug_rotation_num):
                            rotate_theta = float(rotate_theta_random[roi])
                            crop_mask_R = subimage(
                                Mask_image_padding,
                                center_trans,
                                rotate_theta,
                                int(patch_size * scale_ratio),
                                int(patch_size * scale_ratio),
                            )
                            crop_mask_R = crop_mask_R[:, :, 0]
                            crop_mask_name = (
                                save_dir_mask2
                                + "/"
                                + str(case_num)
                                + "_CTR_"
                                + str(i)
                                + "_"
                                + str(roi)
                                + "_"
                                + str(crop_num)
                                + "_"
                                + str(tri)
                                + ".jpg"
                            )
                            cv2.imwrite(crop_mask_name, crop_mask_R)
                            sub_mask_list.append(crop_mask_R)

                            crop_MSI_image_R = subimage(
                                MSI_image_padding,
                                center_trans,
                                rotate_theta,
                                int(patch_size * scale_ratio),
                                int(patch_size * scale_ratio),
                            )
                            crop_MSI_name = (
                                save_dir_image2
                                + "/"
                                + str(case_num)
                                + "_CTR_"
                                + str(i)
                                + "_"
                                + str(roi)
                                + "_"
                                + str(crop_num)
                                + "_"
                                + str(tri)
                                + ".jpg"
                            )
                            cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                            sub_img_list.append(crop_MSI_image_R)
                print("Iteration:")
        crop_num = crop_num + 1

    return img_list, mask_list, sub_img_list, sub_mask_list


def GenTrainImages_negative(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    step=10,
    scale_ratio=-1,
):

    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []

    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    print("save_dir_image:", save_dir_image)
    print("save_dir_mask:", save_dir_mask)

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )

    size_1 = int(patch_size * 1.5)
    size_2 = int((patch_size * scale_ratio - patch_size) / 2)
    ##########################
    height = Mask_image.shape[0]
    width = Mask_image.shape[1]

    w = patch_size
    h = patch_size

    W = int(patch_size * scale_ratio)
    H = int(patch_size * scale_ratio)

    for row in range(0, height - patch_size, step):
        for col in range(0, width - patch_size, step):
            x = col
            y = row

            if (row + patch_size) > height:
                y = height - patch_size
            if (col + patch_size) > width:
                x = width - patch_size

            crop_mask = Mask_image[y : y + h, x : x + w]  # 先用y确定高，再用x确定宽

            flag = 0
            for sub_row in range(0, patch_size):
                for sub_col in range(0, patch_size):
                    if crop_mask[sub_row, sub_col, 0] >= 200:
                        flag = 1
                        break
            print("flag:", flag)
            if flag == 0:
                crop_mask = Mask_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1), 0
                ]  # 先用y确定高，再用x确定宽
                crop_mask_name = (
                    save_dir_mask
                    + "/"
                    + str(case_num)
                    + "_Neg_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                cv2.imwrite(crop_mask_name, crop_mask)
                mask_list.append(crop_mask)
                print("save1:")

                crop_MSI_image = MSI_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1)
                ]  # 先用y确定高，再用x确定宽
                crop_MSI_image_name = (
                    save_dir_image
                    + "/"
                    + str(case_num)
                    + "_Neg_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                print("save2:")
                cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                img_list.append(crop_MSI_image)
                print("2:")

                if scale_ratio > -1:
                    crop_mask = Mask_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                        0,
                    ]  # 先用y确定高，再用x确定宽
                    crop_mask_name = (
                        save_dir_mask2
                        + "/"
                        + str(case_num)
                        + "_Neg_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save3:")
                    cv2.imwrite(crop_mask_name, crop_mask)
                    sub_mask_list.append(crop_mask)

                    crop_MSI_image = MSI_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                    ]  # 先用y确定高，再用x确定宽
                    crop_MSI_image_name = (
                        save_dir_image2
                        + "/"
                        + str(case_num)
                        + "_Neg_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save4:")
                    cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                    sub_img_list.append(crop_MSI_image)

    return img_list, mask_list, sub_img_list, sub_mask_list


def GenTrainImages_augmentation(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    scale_ratio,
    padding_num,
):
    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []
    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    height = Mask_image.shape[0]
    width = Mask_image.shape[1]

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(height / 2),
        int(height / 2),
        int(height / 2),
        int(height / 2),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(height / 2),
        int(height / 2),
        int(height / 2),
        int(height / 2),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    Height = Mask_image_padding.shape[0]
    Width = Mask_image_padding.shape[1]

    ############################ save original image
    if padding_num > 0:
        crop_mask = Mask_image_padding[
            (int(Height / 2) - int(height / 2) - padding_num) : (
                int(Height / 2) + int(height / 2) + padding_num
            ),
            (int(Width / 2) - int(width / 2) - padding_num) : (
                int(Width / 2) + int(width / 2) + padding_num
            ),
            0,
        ]
        crop_MSI_image = MSI_image_padding[
            (int(Height / 2) - int(height / 2) - padding_num) : (
                int(Height / 2) + int(height / 2) + padding_num
            ),
            (int(Width / 2) - int(width / 2) - padding_num) : (
                int(Width / 2) + int(width / 2) + padding_num
            ),
        ]

    else:
        crop_mask = Mask_image_padding[
            (int(Height / 2) - int(height / 2)) : (int(Height / 2) + int(height / 2)),
            (int(Width / 2) - int(width / 2)) : (int(Width / 2) + int(width / 2)),
            0,
        ]
        crop_MSI_image = MSI_image_padding[
            (int(Height / 2) - int(height / 2)) : (int(Height / 2) + int(height / 2)),
            (int(Width / 2) - int(width / 2)) : (int(Width / 2) + int(width / 2)),
        ]

    crop_mask_name = save_dir_mask + "/" + str(case_num) + "_ori_" + str(0) + ".jpg"
    cv2.imwrite(crop_mask_name, crop_mask)
    mask_list.append(crop_mask)
    print("crop_mask:", crop_mask.shape)
    crop_MSI_name = save_dir_image + "/" + str(case_num) + "_ori_" + str(0) + ".jpg"
    cv2.imwrite(crop_MSI_name, crop_MSI_image)
    img_list.append(crop_MSI_image)

    ############################ save rotation image
    if aug_rotation == "yes":
        center = [int(Height / 2), int(Width / 2)]
        size = int(height + padding_num * 2)
        rotate_theta_random = random.sample(
            range(rotation_theta, 360, rotation_theta), aug_rotation_num
        )
        for roi in range(aug_rotation_num):
            rotate_theta = float(rotate_theta_random[roi])
            crop_mask_R = subimage(
                Mask_image_padding[:, :, 0], center, rotate_theta, size, size
            )
            crop_mask_name = (
                save_dir_mask + "/" + str(case_num) + "_ori_" + str(roi + 1) + ".jpg"
            )
            cv2.imwrite(crop_mask_name, crop_mask_R)
            mask_list.append(crop_mask_R)

            crop_MSI_image_R = subimage(
                MSI_image_padding, center, rotate_theta, size, size
            )
            crop_MSI_name = (
                save_dir_image + "/" + str(case_num) + "_ori_" + str(roi + 1) + ".jpg"
            )

            print("crop_mask_R:", crop_mask_R.shape)
            cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
            img_list.append(crop_MSI_image_R)
    return img_list, mask_list, sub_img_list, sub_mask_list


def GenTrainImages_positive_center(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    step=10,
    scale_ratio=-1,
):
    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []
    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )

    size_1 = int(patch_size * 1.5)
    size_2 = int((patch_size * scale_ratio - patch_size) / 2)
    ##########################
    height = Mask_image.shape[0]
    width = Mask_image.shape[1]

    w = patch_size
    h = patch_size

    W = int(patch_size * scale_ratio)
    H = int(patch_size * scale_ratio)

    for row in range(0, height - patch_size, step):
        for col in range(0, width - patch_size, step):
            x = col
            y = row

            if (row + patch_size) > height:
                y = height - patch_size
            if (col + patch_size) > width:
                x = width - patch_size

            crop_mask = Mask_image[y : y + h, x : x + w]  # 先用y确定高，再用x确定宽

            flag = 0
            center_row = int(patch_size / 2)
            center_col = int(patch_size / 2)
            if (
                crop_mask[center_row - 1, center_col - 1, 0] >= 200
                or crop_mask[center_row, center_col - 1, 0] >= 200
                or crop_mask[center_row - 1, center_col, 0] >= 200
                or crop_mask[center_row, center_col, 0] >= 200
            ):
                flag = 1
            # for sub_row in range(0, patch_size):
            #     for sub_col in range(0, patch_size):
            #         if crop_mask[sub_row, sub_col, 0] >= 200:
            #             flag = 1
            #             break
            # print('flag:', flag)
            if flag == 1:
                crop_mask = Mask_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1), 0
                ]  # 先用y确定高，再用x确定宽
                crop_mask_name = (
                    save_dir_mask
                    + "/"
                    + str(case_num)
                    + "_Cen_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                cv2.imwrite(crop_mask_name, crop_mask)
                mask_list.append(crop_mask)
                print("save1:")

                crop_MSI_image = MSI_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1)
                ]  # 先用y确定高，再用x确定宽
                crop_MSI_image_name = (
                    save_dir_image
                    + "/"
                    + str(case_num)
                    + "_Cen_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                print("save2:")
                cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                img_list.append(crop_MSI_image)
                print("2:")

                ############################ crop centered image and rotate
                if aug_rotation == "yes":
                    center = [int(x + size_1 + w / 2), int(y + size_1 + h / 2)]
                    rotate_theta_random = random.sample(
                        range(rotation_theta, 360, rotation_theta), aug_rotation_num
                    )
                    for roi in range(aug_rotation_num):
                        rotate_theta = float(rotate_theta_random[roi])
                        crop_mask_R = subimage(
                            Mask_image_padding,
                            center,
                            rotate_theta,
                            patch_size,
                            patch_size,
                        )
                        crop_mask_R = crop_mask_R[:, :, 0]
                        crop_mask_name = (
                            save_dir_mask
                            + "/"
                            + str(case_num)
                            + "_CenR_"
                            + str(row)
                            + "_"
                            + str(col)
                            + ".jpg"
                        )
                        cv2.imwrite(crop_mask_name, crop_mask_R)
                        mask_list.append(crop_mask_R)
                        crop_MSI_image_R = subimage(
                            MSI_image_padding,
                            center,
                            rotate_theta,
                            patch_size,
                            patch_size,
                        )
                        crop_MSI_name = (
                            save_dir_image
                            + "/"
                            + str(case_num)
                            + "_CenR_"
                            + str(row)
                            + "_"
                            + str(col)
                            + ".jpg"
                        )
                        cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                        img_list.append(crop_MSI_image_R)

                if scale_ratio > -1:
                    crop_mask = Mask_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                        0,
                    ]  # 先用y确定高，再用x确定宽
                    crop_mask_name = (
                        save_dir_mask2
                        + "/"
                        + str(case_num)
                        + "_Cen_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save3:")
                    cv2.imwrite(crop_mask_name, crop_mask)
                    sub_mask_list.append(crop_mask)

                    crop_MSI_image = MSI_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                    ]  # 先用y确定高，再用x确定宽
                    crop_MSI_image_name = (
                        save_dir_image2
                        + "/"
                        + str(case_num)
                        + "_Cen_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save4:")
                    cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                    sub_img_list.append(crop_MSI_image)

                    if aug_rotation == "yes":
                        # rotate_theta_random = random.sample(range(rotation_theta, 360, rotation_theta),
                        #                                     aug_rotation_num)
                        center = [
                            int(x + size_1 + W / 2 - size_2),
                            int(y + size_1 + H / 2 - size_2),
                        ]
                        for roi in range(aug_rotation_num):
                            rotate_theta = float(rotate_theta_random[roi])
                            crop_mask_R = subimage(
                                Mask_image_padding,
                                center,
                                rotate_theta,
                                int(patch_size * scale_ratio),
                                int(patch_size * scale_ratio),
                            )
                            crop_mask_R = crop_mask_R[:, :, 0]
                            crop_mask_name = (
                                save_dir_mask2
                                + "/"
                                + str(case_num)
                                + "_CenR_"
                                + str(row)
                                + "_"
                                + str(col)
                                + ".jpg"
                            )
                            cv2.imwrite(crop_mask_name, crop_mask_R)
                            sub_mask_list.append(crop_mask_R)

                            crop_MSI_image_R = subimage(
                                MSI_image_padding,
                                center,
                                rotate_theta,
                                int(patch_size * scale_ratio),
                                int(patch_size * scale_ratio),
                            )
                            crop_MSI_name = (
                                save_dir_image2
                                + "/"
                                + str(case_num)
                                + "_CenR_"
                                + str(row)
                                + "_"
                                + str(col)
                                + ".jpg"
                            )
                            cv2.imwrite(crop_MSI_name, crop_MSI_image_R)
                            sub_img_list.append(crop_MSI_image_R)
    return img_list, mask_list, sub_img_list, sub_mask_list


def GenTrainImages_negative_notcenter(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    step=10,
    scale_ratio=-1,
):

    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []
    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    print("save_dir_image:", save_dir_image)
    print("save_dir_mask:", save_dir_mask)

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )

    size_1 = int(patch_size * 1.5)
    size_2 = int((patch_size * scale_ratio - patch_size) / 2)
    ##########################
    height = Mask_image.shape[0]
    width = Mask_image.shape[1]

    w = patch_size
    h = patch_size

    W = int(patch_size * scale_ratio)
    H = int(patch_size * scale_ratio)

    for row in range(0, height - patch_size, step):
        for col in range(0, width - patch_size, step):
            x = col
            y = row
            print("x:", x)
            print("y:", y)

            if (row + patch_size) > height:
                y = height - patch_size
            if (col + patch_size) > width:
                x = width - patch_size

            crop_mask = Mask_image[y : y + h, x : x + w]  # 先用y确定高，再用x确定宽

            flag = 1
            sub_row = int(h / 2)
            sub_col = int(w / 2)

            print("sub_row:", sub_row)
            print("sub_col:", sub_col)

            if (
                crop_mask[sub_row, sub_col, 0] < 100
                and crop_mask[sub_row - 1, sub_col, 0] < 100
                and crop_mask[sub_row, sub_col - 1, 0] < 100
                and crop_mask[sub_row - 1, sub_col - 1, 0] < 100
            ):
                flag = 0
            # for sub_row in range(0,patch_size):
            #     for sub_col in range(0,patch_size):
            #         if crop_mask[sub_row,sub_col,0] >=200:
            #             flag = 1
            #             break
            print("flag:", flag)
            if flag == 0:
                crop_mask = Mask_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1), 0
                ]  # 先用y确定高，再用x确定宽
                crop_mask_name = (
                    save_dir_mask
                    + "/"
                    + str(case_num)
                    + "_Neg_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                cv2.imwrite(crop_mask_name, crop_mask)
                mask_list.append(crop_mask)
                print("save1:")

                crop_MSI_image = MSI_image_padding[
                    (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1)
                ]  # 先用y确定高，再用x确定宽
                crop_MSI_image_name = (
                    save_dir_image
                    + "/"
                    + str(case_num)
                    + "_Neg_"
                    + str(row)
                    + "_"
                    + str(col)
                    + ".jpg"
                )
                print("save2:")
                cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                img_list.append(crop_MSI_image)
                print("2:")

                if scale_ratio > -1:
                    crop_mask = Mask_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                        0,
                    ]  # 先用y确定高，再用x确定宽
                    crop_mask_name = (
                        save_dir_mask2
                        + "/"
                        + str(case_num)
                        + "_Neg_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save3:")
                    cv2.imwrite(crop_mask_name, crop_mask)
                    sub_mask_list.append(crop_mask)

                    crop_MSI_image = MSI_image_padding[
                        (y - size_2 + size_1) : (y - size_2 + H + size_1),
                        (x - size_2 + size_1) : (x - size_2 + W + size_1),
                    ]  # 先用y确定高，再用x确定宽
                    crop_MSI_image_name = (
                        save_dir_image2
                        + "/"
                        + str(case_num)
                        + "_Neg_"
                        + str(row)
                        + "_"
                        + str(col)
                        + ".jpg"
                    )
                    print("save4:")
                    cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
                    sub_img_list.append(crop_MSI_image)

    return img_list, mask_list, sub_img_list, sub_mask_list


def GenTrainImages_testpatch(
    MSI_image,
    Mask_image,
    MSI_image_save_file,
    PosOrNeg,
    case_num,
    patch_size,
    step=10,
    scale_ratio=-1,
):
    img_list = []
    mask_list = []
    sub_img_list = []
    sub_mask_list = []
    if case_num < 10:
        str_casenum = "00" + str(case_num)

    elif case_num >= 10 and case_num < 100:
        str_casenum = "0" + str(case_num)

    else:
        str_casenum = str(case_num)

    step = patch_size
    save_dir_image, save_dir_image2 = SaveImageName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )
    save_dir_mask, save_dir_mask2 = SaveMaskName(
        MSI_image_save_file, PosOrNeg, patch_size, scale_ratio
    )

    print("save_dir_image:", save_dir_image)
    print("save_dir_mask:", save_dir_mask)

    BLACK = [0, 0, 0]
    Mask_image_padding = cv2.copyMakeBorder(
        Mask_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )
    MSI_image_padding = cv2.copyMakeBorder(
        MSI_image,
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        int(patch_size * 1.5),
        cv2.BORDER_CONSTANT,
        value=BLACK,
    )

    size_1 = int(patch_size * 1.5)
    size_2 = int((patch_size * scale_ratio - patch_size) / 2)
    ##########################
    height = Mask_image.shape[0]
    width = Mask_image.shape[1]

    w = patch_size
    h = patch_size

    patch_num = 0
    for row in range(0, height, step):
        for col in range(0, width, step):
            patch_num = patch_num + 1
            y = col
            x = row

            crop_mask = Mask_image_padding[
                (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1), 0
            ]  # 先用y确定高，再用x确定宽
            crop_mask_name = (
                save_dir_mask + "/" + str_casenum + "_" + str(patch_num) + ".jpg"
            )
            cv2.imwrite(crop_mask_name, crop_mask)
            mask_list.append(crop_mask)
            print("save1:")

            crop_MSI_image = MSI_image_padding[
                (y + size_1) : (y + h + size_1), (x + size_1) : (x + w + size_1)
            ]  # 先用y确定高，再用x确定宽
            crop_MSI_image_name = (
                save_dir_image + "/" + str_casenum + "_" + str(patch_num) + ".jpg"
            )
            print("save2:")
            cv2.imwrite(crop_MSI_image_name, crop_MSI_image)
            img_list.append(crop_MSI_image)


# https://www.jianshu.com/p/b87a5408d408
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


def CreateSaveFile(MSI_image_save_file, PosOrNeg, patch_size, scale_ratio):

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
