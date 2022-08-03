import os
import sys
import cv2
from matplotlib import pyplot as plt
from classes.Anno_xml2list import Anno_xml2list
from classes.make_datapath_list import make_datapath_list
from pathlib import Path

from classes.DataTransform import DataTransform
from classes.VOCDataset import VOCDataset
from classes.Dataloader import VOCDataLoader

rootpath = os.path.join(Path.cwd().parent.absolute(), "data/VOCdevkit/VOC2012")

train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath
)


voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

transform_anno = Anno_xml2list(classes=voc_classes)

index = 1
image_file_path = val_img_list[index]
print(image_file_path)
img = cv2.imread(image_file_path)
height, width, channels = img.shape

# アノテーションをリスト化
anno_list = transform_anno(val_anno_list[index], width, height)

# 原画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# VOCDatasetの動作確認
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(
    train_img_list,
    train_anno_list,
    phase="train",
    transform=DataTransform(
        input_size, color_mean
    ),
    transform_anno=Anno_xml2list(voc_classes)
)

val_dataset = VOCDataset(
    val_img_list,
    val_anno_list,
    phase="val",
    transform=DataTransform(input_size, color_mean),
    transform_anno=Anno_xml2list(voc_classes)
)

print(val_dataset.__getitem__(1))

# データローダーの作成
batch_size = 4
train_dataloader = VOCDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_dataloader = VOCDataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
dataloader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}

# 動作確認
batch_iterator = iter(dataloader_dict["val"])  # イテレータに変換
images, targets = next(batch_iterator)  # 一番目の要素を取り出す
print(images.size())
print(len(targets))
print(targets[1].size())

print(train_dataset.__len__())
print(val_dataset.__len__())
