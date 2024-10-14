import os
import pickle
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import cv2
from torch.utils import data

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    with open('../xgw/segmentation/config.json') as f:
        js = f.read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

def getDatasets(dir):
    return os.listdir(dir)

def resize_image(origin_img, long_edge=1024, short_edge=960):
    if origin_img is None or origin_img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    im_lr = origin_img.shape[0]
    im_ud = origin_img.shape[1]
    new_img = np.zeros([long_edge, short_edge, 3], dtype=np.uint8)
    new_shape = new_img.shape[:2]

    if im_lr > im_ud:
        img_shrink, base_img_shrink = long_edge, long_edge
        im_ud = int(im_ud / im_lr * base_img_shrink)
        im_ud += 32 - im_ud % 32
        im_ud = min(im_ud, short_edge)
        im_lr = img_shrink
        origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
        new_img[:, (new_shape[1] - im_ud) // 2:new_shape[1] - (new_shape[1] - im_ud) // 2] = origin_img
    else:
        img_shrink, base_img_shrink = short_edge, short_edge
        im_lr = int(im_lr / im_ud * base_img_shrink)
        im_lr += 32 - im_lr % 32
        im_lr = min(im_lr, long_edge)
        im_ud = img_shrink
        origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
        new_img[(new_shape[0] - im_lr) // 2:new_shape[0] - (new_shape[0] - im_lr) // 2, :] = origin_img

    return new_img

class PerturbedDatastsForFiducialPoints_pickle_color_v2_v2(data.Dataset):
    def __init__(self, root, split='1-1', img_shrink=None, is_return_img_name=False, preproccess=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_shrink = img_shrink
        self.is_return_img_name = is_return_img_name
        self.preproccess = preproccess
        self.images = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.row_gap = 1
        self.col_gap = 1
        datasets = ['validate', 'test', 'train']

        if self.split == 'test' or self.split == 'eval':
            img_file_list = getDatasets(os.path.join(self.root))
            self.images[self.split] = img_file_list
        elif self.split in datasets:
            img_file_list = []
            img_file_list_ = getDatasets(os.path.join(self.root, 'color'))
            for id_ in img_file_list_:
                img_file_list.append(id_.rstrip())

            self.images[self.split] = sorted(img_file_list, key=lambda num: (
                re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(1),
                int(re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(2)),
                int(re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(3)),
                re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(4)))
        else:
            raise Exception('load data error')

    def checkImg(self):
        if self.split == 'validate':
            for im_name in self.images[self.split]:
                im_path = pjoin(self.root, self.split, 'color', im_name)
                try:
                    with open(im_path, 'rb') as f:
                        perturbed_data = pickle.load(f)
                    im_shape = perturbed_data.shape
                except Exception as e:
                    print(f"Error loading image {im_name}: {e}")

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, item):
        im_name = self.images[self.split][item]
        im_path = pjoin(self.root, im_name)

        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image path does not exist: {im_path}")

        im = cv2.imread(im_path, flags=cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Image could not be read from path: {im_path}")

        im = self.resize_im(im)
        im = self.transform_im(im)

        if self.is_return_img_name:
            return im, im_name
        return im

    def transform_im(self, im):
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()
        return im

    def resize_im(self, im):
        if im is None or im.size == 0:
            raise ValueError("Image is empty or invalid.")
        im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
        return im

    def resize_lbl(self, lbl):
        lbl = lbl / [960, 1024] * [992, 992]
        return lbl

    # def resize_lbl(self, lbl, original_image_shape, resized_image_shape):
    #     scale_x = resized_image_shape[1] / original_image_shape[1]  # Tỉ lệ theo chiều ngang
    #     scale_y = resized_image_shape[0] / original_image_shape[0]  # Tỉ lệ theo chiều dọc
    #     lbl = lbl * [scale_x, scale_y]  # Áp dụng tỉ lệ thay đổi lên nhãn
    #     return lbl

    def fiducal_points_lbl(self, fiducial_points, segment):
        fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
        fiducial_points = fiducial_points[::fiducial_point_gaps[self.row_gap], ::fiducial_point_gaps[self.col_gap], :]
        segment = segment * [fiducial_point_gaps[self.col_gap], fiducial_point_gaps[self.row_gap]]
        return fiducial_points, segment
