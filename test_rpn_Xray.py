import pickle
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import roi_helpers
import resnet as nn
import os
import time
import cv2
import numpy as np
from visual_rpn import visual_rpn
import argparse


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.img_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio, new_width, new_height


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio, new_width, new_height = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio, new_width, new_height


def predict_single_image(img_path, model_rpn, cfg, class_mapping):
    st = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)

    X, ratio, new_width, new_height = format_img(img, cfg)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    # for x in range(Y1.shape[1]):
    #     for y in range(Y1.shape[2]):
    #         for z in range(Y1.shape[3]):
    #             print(Y1[0][x][y][z])
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.9)

    visual_rpn(img_path, new_width, new_height, result)


def predict(args_):
    path = args_.path
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # 定义RPN网络结构
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    model_rpn = Model(img_input, rpn_layers)
    print('从{}加载权重'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_rpn.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', 'jpeg', 'jpg', '.png', 'tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn, cfg, class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, cfg, class_mapping)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='/home/nice/Pictures/Xray_metalcup/', help='image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)