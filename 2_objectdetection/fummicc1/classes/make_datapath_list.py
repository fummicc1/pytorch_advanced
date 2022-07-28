import os
from typing import List, Tuple


def make_datapath_list(rootpath: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """データへのパスを格納したリストを作成
    """

    imgpath_template = os.path.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = os.path.join(rootpath, "Annotations", "%s.xml")

    # 訓練と検証, それぞれのファイルのID（ファイル名）を取得する
    train_id_names = os.path.join(rootpath, "ImageSets/Main/train.txt")
    val_id_names = os.path.join(rootpath, "ImageSets/Main/val.txt")

    train_img_list = []
    train_anno_list = []

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = []
    val_anno_list = []

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list
