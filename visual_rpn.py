from config import Config
import cv2
import numpy as np
import os


def visual_rpn(filename, resized_width, resized_height, result):
    # assert result.shape[0] == 10
    assert result.shape[1] == 4
    img = cv2.imread(filename)
    height, width = img.shape[:2]
    cfg = Config()
    x_factor = (resized_width / float(width)) / cfg.rpn_stride
    y_factor = (resized_height / float(height)) / cfg.rpn_stride
    bboxes = np.zeros((10, 4))
    for idx in range(result.shape[0]):
        x1 = result[idx, 0]
        y1 = result[idx, 1]
        x2 = result[idx, 2]
        y2 = result[idx, 3]
        x1_origin = x1 / x_factor
        y1_origin = y1 / y_factor
        x2_origin = x2 / x_factor
        y2_origin = y2 / y_factor
        bboxes[idx, 0] = x1_origin
        bboxes[idx, 1] = y1_origin
        bboxes[idx, 2] = x2_origin
        bboxes[idx, 3] = y2_origin
    visual_img = draw_boxes_and_label_on_image_cv2(img, bboxes)
    result_path = './results_images/{}.bmp'.format(os.path.basename(filename).split('.')[0])
    print('resule saved into ', result_path)
    cv2.imwrite(result_path, visual_img)


def draw_boxes_and_label_on_image_cv2(img, bboxes):
    for idx in range(bboxes.shape[0]):
        print(idx)
        bb_left = int(bboxes[idx][0])
        bb_top = int(bboxes[idx][1])
        bb_width = int(bboxes[idx][2])
        bb_height = int(bboxes[idx][3])
        img = cv2.rectangle(img, (bb_left, bb_top), (bb_width, bb_height), (int(idx * 20), int(idx * 20), int(idx * 20)), 2)
    return img

