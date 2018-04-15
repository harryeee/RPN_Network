import cv2
import numpy as np
import copy


def augment(img_data, config, augment=True):
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug.filename)

    if augment:
        rows, cols = img.shape[:2]

        # 50%的概率翻转图像
        # 是否水平翻转
        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            pass

        # 是否垂直翻转
        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            pass

        # 是否旋转90度
        if config.rot_90:
            pass

    # print(" before ")
    # print(img_data_aug.width)
    img_data_aug.width = img.shape[1]
    img_data_aug.height = img.shape[0]
    # print(" after ")
    # print(img_data_aug.width)

    return img_data_aug, img
