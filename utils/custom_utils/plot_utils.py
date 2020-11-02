import os
import cv2


def plotresult(bg_img_path, manual_img_path, predict_img_path, dst_path):

    img = cv2.imread(bg_img_path)

    mask_img = cv2.imread(manual_img_path, 0)  # gray
    ret1, mask_thresh1 = cv2.threshold(mask_img, 100, 255, cv2.THRESH_BINARY)
    _, contours1, hierarchy1 = cv2.findContours(
        mask_thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    predict_img = cv2.imread(predict_img_path, 0)
    ret3, mask_thresh1_predcit = cv2.threshold(predict_img, 100, 255, cv2.THRESH_BINARY)
    _, contours2, hierarchy2 = cv2.findContours(
        mask_thresh1_predcit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # loop over the contours
    for c in contours1:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = int(M["m10"])
            cY = int(M["m01"])

        if cX == cY == 0:
            cX = c[0][0][0]
            cY = c[0][0][1]
        cv2.circle(
            img, (cX, cY), 12, (255, 0, 0), 2
        )  # -1 means fullfill circle, candidate blue

    for c2 in contours2:
        # print(c2)
        M = cv2.moments(c2)
        # print(M)
        if M["m00"] != 0:
            cX2 = int(M["m10"] / M["m00"])
            cY2 = int(M["m01"] / M["m00"])
        else:
            cX2 = int(M["m10"])
            cY2 = int(M["m01"])

        # print(cX2)
        # print(cY2)
        if cX2 == cY2 == 0:
            cX2 = c2[0][0][0]
            cY2 = c2[0][0][1]

        # print(cX2)
        # print(cY2)
        cv2.circle(img, (cX2, cY2), 20, (0, 0, 255), 2)  # green manual

    cv2.imwrite(dst_path, img)
