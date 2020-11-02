###################################################
#
#   Script to pre-process the original imgs
#
##################################################
# channel last

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    assert len(rgb.shape) == 4  # 4D arrays
    assert rgb.shape[3] == 3
    bn_imgs = (
        rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114
    )
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return bn_imgs


# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert data.shape[3] == 1 or data.shape[3] == 3
    # data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


#
# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert len(data.shape) == 3  # height*width*channels
    img = None
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + ".png")
    return img


# prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert len(masks.shape) == 4  # 4D arrays
    assert masks.shape[3] == 1  # check the channel is 1
    im_h = masks.shape[1]
    im_w = masks.shape[2]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i, j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


def masks_Unet2(masks, num_lesion_class):
    assert len(masks.shape) == 4  # 4D arrays
    # assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], num_lesion_class, im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, num_lesion_class + 1))

    new_masks[:, :, 0:num_lesion_class] = masks[:, 0:num_lesion_class, :].transpose(
        0, 2, 1
    )
    # for i in range(4):
    mask0 = new_masks[:, :, 0]
    m0 = np.ma.array(new_masks[:, :, 0], mask=mask0)

    # mask1=new_masks[:,:,1]
    # m1=np.ma.array(new_masks[:,:,1],mask=mask1)
    #
    # mask2=new_masks[:,:,2]
    # m2=np.ma.array(new_masks[:,:,2],mask=mask2)
    #
    # mask3=new_masks[:,:,3]
    # m3=np.ma.array(new_masks[:,:,3],mask=mask3)

    new_masks[:, :, num_lesion_class] = 1 - (m0.mask)  # |m1.mask|m2.mask|m3.mask)
    # for i in range(masks.shape[0]):
    #    for j in range(im_h*im_w):
    #        cnt=0
    #        for k in range(4):
    #            if  masks[i,k,j] == 0:
    #                new_masks[i,j,k]=0
    #                cnt=cnt+1
    #            else:
    #                new_masks[i,j,k]=1
    #        if cnt==4:
    #            new_masks[i,j,4]=1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert len(pred.shape) == 3  # 3D array: (Npatches,height*width,2)
    assert pred.shape[2] == 2  # check the classes are 2
    pred_images = np.empty((pred.shape[0], pred.shape[1]))  # (Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print(
            "mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'"
        )
        exit()
    pred_images = np.reshape(
        pred_images, (pred_images.shape[0], patch_height, patch_width, 1)
    )
    print("pred_images:", pred_images.shape)
    return pred_images


def pred_to_imgs2(pred, patch_height, patch_width, num_lesion_class, mode="original"):
    assert len(pred.shape) == 3  # 3D array: (Npatches,height*width,2)
    # assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = None  # (Npatches,height*width)
    if mode == "original":
        pred_images = pred[:, :, 0:2]
        # pred_images[:,:,1]=pred[:,:,3]
        # for i in range(pred.shape[0]):
        #    for pix in range(pred.shape[1]):
        #        if pred[i,pix,3]>0.5:
        #            pred_images[i,pix,0:3]=1.0
        # pred_images[i,pix]=pred_images[i,pix]+pred[i,pix,3]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print(
            "mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'"
        )
        exit()
    pred_images = np.reshape(
        pred_images, (pred_images.shape[0], patch_height, patch_width, 2)
    )
    return pred_images


def my_PreProc2(data):
    assert len(data.shape) == 4
    assert data.shape[1] == 1  # Use the original images
    # black-white conversion
    # train_imgs = rgb2gray(data)
    train_imgs = np.zeros(data.shape)
    train_imgs0 = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
    # train_imgs1=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
    # train_imgs2=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
    train_imgs0[:, 0, :, :] = data[:, 0, :, :]
    # train_imgs1[:,0,:,:]=data[:,1,:,:]
    # train_imgs2[:,0,:,:]=data[:,2,:,:]

    # train_imgs = data
    # my preprocessing:
    #    train_imgs = dataset_normalized(train_imgs)
    #    train_imgs = clahe_equalized(train_imgs)
    #    train_imgs = adjust_gamma(train_imgs, 1.2)
    #    train_imgs = train_imgs/255.  #reduce to 0-1 range
    train_imgs0 = dataset_normalized2(train_imgs0)
    train_imgs0 = clahe_equalized2(train_imgs0)
    train_imgs0 = adjust_gamma2(train_imgs0, 1.2)
    train_imgs0 = train_imgs0 / 255.0  # reduce to 0-1 range
    # train_imgs1 = dataset_normalized(train_imgs1)
    # train_imgs1 = clahe_equalized(train_imgs1)
    # train_imgs1 = adjust_gamma(train_imgs1, 1.2)
    # train_imgs1 = train_imgs1/255.  #reduce to 0-1 range
    # train_imgs2 = dataset_normalized(train_imgs2)
    # train_imgs2 = clahe_equalized(train_imgs2)
    # train_imgs2 = adjust_gamma(train_imgs2, 1.2)
    # train_imgs2 = train_imgs2/255.  #reduce to 0-1 range
    train_imgs[:, 0, :, :] = train_imgs0[:, 0, :, :]
    # train_imgs[:,1,:,:]=train_imgs1[:,0,:,:]
    # train_imgs[:,2,:,:]=train_imgs2[:,0,:,:]
    return train_imgs


# My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert len(data.shape) == 4
    assert data.shape[3] == 3  # Use the original images
    # black-white conversion
    train_imgs = rgb2gray(data)
    # my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.0  # reduce to 0-1 range
    return train_imgs


# ============================================================
# ========= PRE PROCESSING FUNCTIONS ========================#
# ============================================================

# ==== histogram equalization
def histo_equalized(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[3] == 1  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = cv2.equalizeHist(
            np.array(imgs[i, :, :, 0], dtype=np.uint8)
        )
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[3] == 1  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = clahe.apply(
            np.array(imgs[i, :, :, 0], dtype=np.uint8)
        )
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[3] == 1  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (
            (imgs_normalized[i] - np.min(imgs_normalized[i]))
            / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))
        ) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[3] == 1  # check the channel is 3
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, :, :, 0] = cv2.LUT(
            np.array(imgs[i, :, :, 0], dtype=np.uint8), table
        )
    return new_imgs


def histo_equalized2(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized2(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[1] == 1  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized2(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[1] == 1  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (
            (imgs_normalized[i] - np.min(imgs_normalized[i]))
            / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))
        ) * 255
    return imgs_normalized


def adjust_gamma2(imgs, gamma=1.0):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[1] == 1  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def get_all_points_from_mask(mask):
    ret1, mask_thresh1 = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    idx = mask_thresh1 > 100
    points = np.transpose(np.nonzero(idx))
    return points


def get_center_points_from_mask(mask):

    ret1, mask_thresh1 = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    # plt.imshow(mask_thresh1,'gray')
    # plt.show()

    _, contours1, hierarchy1 = cv2.findContours(
        mask_thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # loop over the contours
    points = []
    num1 = 0
    for c in contours1:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = int(M["m10"])
            cY = int(M["m01"])
        num1 = num1 + 1

        if cX == cY == 0:
            cX = c[0][0][0]
            cY = c[0][0][1]

        l = (cY, cX)  # 注意此处是先x，后y还是其他！！！
        # print("point:",l)
        points.append(l)

    print("-----------------")
    points = np.array(points)
    return points


def generateCircleMask(img, sizedilation=4, threshmin=10):
    thresharea = img.shape[0] * img.shape[1] / 4
    # print('type(img)', type(img))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    ret, threimage = cv2.threshold(img, threshmin, 255.0, cv2.THRESH_BINARY)

    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sizedilation, sizedilation))
    elem1 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(sizedilation / 2), int(sizedilation / 2))
    )

    dilateimage = cv2.dilate(threimage, elem, iterations=4)
    erodeimage = cv2.erode(dilateimage, elem1, iterations=4)
    # erodeimage = cv2.erode(erodeimage, elem, iterations=4)

    img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    _, contours, hierarchy = cv2.findContours(
        erodeimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > thresharea:
            cv2.drawContours(img, contours, 0, 255, -1)
            # print(area)

    _, threimage = cv2.threshold(img, threshmin, 255.0, cv2.THRESH_BINARY)
    # cv2.namedWindow("da")
    # cv2.imshow("da", threimage)
    # cv2.waitKey(0)
    return threimage


def intersection_mask(mask_data):
    assert len(mask_data.shape) == 3
    result_data = np.zeros((mask_data.shape[0], mask_data.shape[1]))
    for i in range(mask_data.shape[2]):
        result_data = result_data + mask_data[:, :, i]
    # print(np.max(result_data))
    # print(np.unique(result_data))
    idx = result_data > (mask_data.shape[2] * 255 - 65)
    # print(idx)
    result_data[idx] = 255
    result_data[~idx] = 0
    return result_data
