import config
import pickle
import pprint
import random
import data_generators
import time
import resnet as nn
import numpy as np
import losses as losses_fn
import roi_helpers as roi_helpers
from get_data import get_data
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import generic_utils


def train_rpn():
    # 读取配置
    cfg = config.Config()

    # 将图像及VOC格式的数据以Img_Data对象的形式进行保存
    all_images, classes_count, class_mapping = get_data(cfg.label_file)
    cfg.class_mapping = class_mapping

    # for bbox_num, bbox in enumerate(all_images[0].bboxes):
    #     print(bbox_num, bbox)

    # 将配置文件进行保存
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('2、Config已经被写入到{}, 并且可以在测试的时候加载以确保得到正确的结果'.format(
            cfg.config_save_file))

    print("3、按照类别数量大小顺序输出")
    pprint.pprint(classes_count)

    print("4、类别个数（大于训练集+测试集数量，并且包括背景）= {}".format(len(classes_count)))

    random.shuffle(all_images)
    print("5、对样本进行打乱")

    train_imgs = [img_data for img_data in all_images if img_data.imageset == 'trainval']
    val_imgs = [img_data for img_data in all_images if img_data.imageset == 'test']
    print("6、设置训练集及验证集，其中训练集数量为{}，测试集数量为{}".format(len(train_imgs), len(val_imgs)))

    # 对训练数据进行打乱
    random.shuffle(train_imgs)

    # 类别映射
    # 得到每一个锚的训练数据，供RPN网络训练使用
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_out_length,
                                                   K.image_dim_ordering(), mode='train')
    # data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,
    #                                              K.image_dim_ordering(), mode='val')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors, cfg.num_regions)
    model_rpn = Model(img_input, rpn[:2])

    try:
        print('7、从{}加载参数'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.model_path, by_name=True)
    except Exception as e:
        print(e)
        print('无法加载与训练模型权重 ')

    optimizer = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])

    epoch_length = 500
    num_epochs = int(cfg.num_epochs)
    iter_num = 0
    losses = np.zeros((epoch_length, 2))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    best_loss = np.Inf

    print('8、开始训练')
    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch{}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('RPN的bounding boxes平均覆盖数量 = {} for {} previous iterations'.format(
                        mean_overlapping_bboxes, epoch_length))

                    if mean_overlapping_bboxes == 0:
                        print('RPN不生产覆盖的边框，检查RPN的设置或者继续训练')

                X, Y, img_data = next(data_gen_train)
                # Y[0].shape (1, X, Y, 18)

                # X是input data，Y是labels
                loss_rpn = model_rpn.train_on_batch(X, Y)

                p_rpn = model_rpn.predict_on_batch(X)

                result = roi_helpers.rpn_to_roi(p_rpn[0], p_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=10)
                # visual_rpn(img_data, result)
                print('-------result------')
                # print('-------result--------')
                # print(result[250])
                # print('-------result--------')
                # print('-------p_rpn--------')
                # print(p_rpn[0].shape)
                # print(p_rpn[1].shape)
                # (1, 38, 48, 9)
                # (1, 38, 48, 36)
                # print('-------p_rpn--------')
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                iter_num += 1

                progbar.update(iter_num, [('rpn分类损失', np.mean(losses[:iter_num, 0])), ('rpn回归损失', np.mean(losses[:iter_num, 1]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    print(loss_rpn_cls, loss_rpn_regr)
                    # mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) // len(rpn_accuracy_for_epoch)
                    # print(mean_overlapping_bboxes)
                    # rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        # print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            ('总损失函数从{}减到{}，保存权重'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_rpn.save_weights(cfg.model_path)

                    break

            except Exception as e:
                print('错误{}'.format(e))
                # 保存模型
                model_rpn.save_weights(cfg.model_path)
                continue
    print('训练完成，退出')


if __name__ == '__main__':
    train_rpn()