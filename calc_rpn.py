import numpy as np
import random


# 计算两个矩形框的iou值
def calc_iou(a, b):
    # 坐标a和坐标b应该是(x1, y1, x2, y2)的格式
    # union：计算两个面积的并
    def union(au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    # intersection：计算两个面积的交
    def intersection(ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = max(ai[2], bi[2]) - x
        h = max(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w * h

    # if语句是要求右下角的点大于左上角的点，属于逻辑检查
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    # 最后返回交并并比 分母加上1e-6，是为了防止分母为0
    return float(area_i) / float(area_u + 1e-6)


def calc_rpn(cfg, img_data, width, height, resized_width, resized_height, img_length_calc_function):

    # 大体算下 600/38=15.78,这里暂时写成16
    downscale = float(cfg.rpn_stride)
    anchor_sizes = cfg.anchor_box_scales
    anchor_ratios = cfg.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    ratios_amount = len(anchor_ratios)
    num_bboxes = len(img_data.bboxes)

    # 计算rpn网络的feature map图尺寸大小
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    # 初始化输出目标
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # 得到ground truth框的坐标，并且由于图像大小改变重新调整其值
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data.bboxes):
        gta[bbox_num, 0] = bbox.x1 * (resized_width / float(width))
        gta[bbox_num, 1] = bbox.x2 * (resized_width / float(width))
        gta[bbox_num, 2] = bbox.y1 * (resized_height / float(height))
        gta[bbox_num, 3] = bbox.y2 * (resized_height / float(height))

    # rpn ground truth
    # 遍历所有的anchor组合
    # [(128, 128), (128, 256), (256, 128), (256, 256), (256, 512), (512, 256), (512, 512), (512, 1024), (1024, 512)]
    # ix和jy是 feature map 上的坐标，而x1_anc, x2_anc等是原图上的坐标
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(ratios_amount):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # 每一个ix都是作为feature中心点映射回原图中心点，超过边框的不计入
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type代表是否anchor是否是目标
                    # 注：现在我们确定了一个预选框组合有确定了中心点那就是唯一确定一个框了，
                    # 接下来就是来确定这个宽的性质了：是否包含物体、如包含物体其回归梯度是多少】
                    # 要确定以上两个性质，每一个框都需要遍历图中的所有bboxes

                    bbox_type = 'neg'
                    best_iou_for_loc = 0.0
                    for bbox_num in range(num_bboxes):
                        curr_iou = calc_iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                            [x1_anc, y1_anc, x2_anc, y2_anc])

                        # 计算regression目标值，如果需要的话
                        # 如果现在的交并比curr_iou大于该bbox最好的交并比或者大于给定的阈值则求下列参数，这些参数是后来要用的即回归梯度
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > cfg.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            # tx：两个框中心的宽的距离与预选框宽的比
                            # ty: 同tx
                            # tw: bbox的宽与预选框宽的比
                            # th: 同理
                            # np.log()表示以e为底的对数

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        # 如果相交的不是背景，那么进行一系列更新
                        if img_data.bboxes[bbox_num] != 'bg':

                            # 关于bbox的相关信息更新
                            # 预选框的相关更新：如果交并比大于阈值这是pos
                            # best_iou_for_loc：其记录的是有最大交并比为多少和其对应的回归梯度
                            # num_anchors_for_bbox[bbox_num]：记录的是bbox拥有的pos预选框的个数
                            # 如果小于最小阈值是neg，在这两个之间是neutral
                            # 需要注意的是：判断一个框为neg需要其与所有的bbox的交并比都小于最小的阈值

                            # 所有的GT框都应该有一个锚点框与之对应，有一个我们认为是best的框
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # 所有IOU > 0.7的都作为pos的bbox
                            if curr_iou > cfg.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1

                                # 更新 regression layer target 如果IOU是当前最好的
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # 如果IOU > 0.3 且 < 0.7 则bbox属于neutral（灰色地带）
                            if cfg.rpn_min_overlap < curr_iou < cfg.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # 当结束对所有的bbox的遍历时，来确定该预选宽的性质。
                    # y_is_box_valid：该预选框是否可用（nertual就是不可用的）
                    # y_rpn_overlap：该预选框是否包含物体
                    # y_rpn_regr:回归梯度
                    # 第三列数据总共是需要9个 num_anchors 然后这个计算方式的话是从0开始到8，其实感觉直接从0也可以
                    # ，只要保证第三列的序不重复，只不过也是另一种表示方法了
                    # 这里是 0 + 3 * 0 = 0, 1 + 3 * 0 = 1, ... , 2 + 3 * 2 = 8
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + ratios_amount * anchor_size_idx] = 1

                        # 确定每个start点的值，总范围是0到num_anchors * 4 = 36
                        start = 4 * (anchor_ratio_idx + ratios_amount * anchor_size_idx)
                        # 0 - 4
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # 确保每个bbox至少有一个positive RPN区域（不包括bg）
    # 如果一个都没有，直接给他把best_anchor_bbox的赋值
    # 还有best_dx_for_bbox
    for idx in range(num_anchors_for_bbox.shape[0]):
        # 只记录bbox > 0.7的，所以如果num_anchors_for_bbox[idx]的值大于0，
        # 那么他必有 positive 的区域
        # 如果 num_anchors_for_bbox[idx]  == 0  说明没有positive的区域，
        # 而如果他的best_anchor_for_bbox == -1 说明他是背景，直接跳过
        # 如果不是背景，则说明他的是neutral或者是neg的情况
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            # iou最低都是0，大于-1，这里应该是bg的情况
            if best_anchor_for_bbox[idx, 0] == -1:
                continue

            # best_anchor_for_bbox[idx, 0] = jy
            # best_anchor_for_bbox[idx, 1] = jx
            # best_anchor_for_bbox[idx, 2] = anchor_ratio_idx
            # best_anchor_for_bbox[idx, 3] = anchor_size_idx
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + ratios_amount *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + ratios_amount *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + ratios_amount * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    # 改变矩阵序列，相当于将anchor的序号放在第一列
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    # 由于新加了一个维度，所以都是np.logical_and
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    # pos_locs是三维数据，因为第一维都是0
    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        # 这里留128个pos，其他的都置为0
        # can't multiply sequence by non-int of type 'float'
        # 如果不把后面那部分转换为int是不可以的
        val_locs = random.sample(range(len(pos_locs[0])), int(len(pos_locs[0]) - num_regions / 2))
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), int(len(neg_locs[0]) - num_pos))
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    # ------y_rpn_cls-------
    # (1, 9, 38, 54)
    # (1, 9, 38, 54)
    # (1, 18, 38, 54)
    # ------y_rpn_cls-------
    # y_is_box_valid 代表是否这个锚点有效，y_rpn_overlap 代表这个锚点是作为pos还是neg
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)

    # ------y_rpn_regr-------
    # (1, 9, 38, 54)
    # (1, 72, 38, 54)
    # ------y_rpn_regr-------
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


if __name__ == '__main__':
    # tee = np.zeros((2, 3, 4))
    # tee_te = np.expand_dims(tee, axis=1)
    # print(tee.shape)
    # print(tee_te.shape)
    list_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    slice_test = random.sample(list_test, 5)
    print(slice_test)
