import os
import sys
import cv2
from classes.Anno_xml2list import Anno_xml2list
from classes.make_datapath_list import make_datapath_list
from pathlib import Path

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

print(transform_anno(val_anno_list[index], width, height))
