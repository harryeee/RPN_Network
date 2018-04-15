import itertools
import data_augment
import numpy as np
from calc_rpn import calc_rpn
import cv2


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False
        for detect_data in img_data.bboxes:
            class_name = detect_data.class_name
            if class_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break
        if class_in_img:
            return False
        else:
            return True


# 将图像相对小的一边调整为600，并改变相关值
def get_new_img(width, height, x_img, img_size=600 ):
    if width <= height:
        f = float(img_size) / width
        resized_height = int(f * height)
        resized_width = img_size
    else:
        f = float(img_size) / height
        resized_width = int(f * width)
        resized_height = img_size
    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    return resized_width, resized_height, x_img


# 获得锚点的ground truth值
def get_anchor_gt(all_img_data, classes_count, cfg, img_length_calc_function, backend, mode='train'):

    sample_selector = SampleSelector(classes_count)
    while True:
        for img_data in all_img_data:
            try:
                if cfg.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, cfg, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, cfg, augment=False)

                (width, height) = (img_data_aug.width, img_data_aug.height)
                (rows, cols, noting) = x_img.shape

                assert cols == width
                assert rows == height

                resized_width, resized_height, x_img = get_new_img(width, height, x_img, cfg.img_size)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(cfg, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                    # print('---------------------')
                    # print(y_rpc_cls, y_rpn_regr)
                except Exception as eor:
                    print(eor)
                    continue

                # Zero-center by mean pixel, and preprocess image
                # 因为opencv读取图片的通道是BGR这里转换为RGB
                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= cfg.img_channel_mean[0]
                x_img[:, :, 1] -= cfg.img_channel_mean[1]
                x_img[:, :, 2] -= cfg.img_channel_mean[2]
                x_img /= cfg.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= cfg.std_scaling

                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
                # 减去均值
                # 将深度变为第一个维度
                # 给图片增加一个维度
                # 给回归梯度除上一个规整因子
                # 如果用的是tf内核,还是要把深度调到最后一位了

            except Exception as e:
                print(e)
                continue
