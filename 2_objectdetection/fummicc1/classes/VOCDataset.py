import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index: int):
        img, anno, _, _ = self.pull_item(index)
        return img, anno

    def pull_item(self, index: int):
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        # 2. xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute((2, 0, 1))
        ground_truth = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, ground_truth, height, width
