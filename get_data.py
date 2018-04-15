import cv2
import numpy as np
from img_data import ImgData, DetectData


def get_data(path):

    found_bg = False
    all_imgs = {}  # 存储 filename 和 img_data的键值对
    classes_count = {}  # 存储各类别的数目, map类型
    class_mapping = {}
    all_data = []  # 存储img_data的列表, list类型

    with open(path, 'r') as f:

        print('0、解析文件中')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            # 对类别进行计数
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('1、发现特殊类别名bg，将作为背景区域（经常用来做负样本挖掘）')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            # 将图像地址载入到all_imgs中
            if filename not in all_imgs:
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                img_data = ImgData(filename, width=cols, height=rows)
                all_imgs[filename] = img_data
                # 随机生成训练和测试样本，按照 5：1的比例
                if np.random.randint(0, 6) > 0:
                    img_data.imageset = 'trainval'
                else:
                    img_data.imageset = 'test'
            detect_data = DetectData(class_name, int(float(x1)), int(float(x2)), int(float(y1)), int(float(y2)))
            all_imgs[filename].bboxes.append(detect_data)

        for filename in all_imgs:
            all_data.append(all_imgs[filename])

        if found_bg:
            pass

        if 'bg' not in classes_count:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)

        return all_data, classes_count, class_mapping
