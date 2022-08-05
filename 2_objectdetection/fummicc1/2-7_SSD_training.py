from mimetypes import init
import os
import sys
import time
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.backends.cudnn import benchmark
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import kaiming_normal_, constant_
from classes.Anno_xml2list import Anno_xml2list
from classes.make_datapath_list import make_datapath_list
from pathlib import Path

from classes.loss import MultiBoxLoss
from classes.DataTransform import DataTransform
from classes.VOCDataset import VOCDataset
from classes.Dataloader import VOCDataLoader
from classes.SSD import SSD

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

# データローダーの作成
batch_size = 64
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

# SSD300の設定
ssd_config = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2, ], [2, 3, ], [2, 3, ], [2, 3, ], [2, ], [2, ]],
}

# SSDネットワークモデル
net = SSD(phase="train", cfg=ssd_config)
# vggの部分は学習済みの重みデータをロードする
vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)

# ssdのその他のネットワークの重みはHeの初期値で初期化


def weights_init(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        kaiming_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias, 0.0)


net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")
print("ネットワーク設定完了: 学習済みの重みをロードしました")

criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=0.3, device=device)

optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)

num_epochs = 50


def train_and_eval():
    global net, dataloader_dict, criterion, optimizer, num_epochs, device
    print(f"使用デバイス: {device}")
    net.to(device)

    # ネットワークがある程度固定であれば高速化させる
    benchmark = True

    # イテレーション
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(1, num_epochs+1):
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-----------')
        print(f'Epoch {epoch}/{num_epochs}')
        print('-----------')

        # epochごとに訓練と検証のループ
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print(" (train) ")
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print('-----------')
                    print(" (val) ")
                else:
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloader_dict[phase]:
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]

                # オプティマイザーの初期化
                optimizer.zero_grad()

                # 順伝搬計算
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward()

                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                        nn.utils.clip_grad.clip_grad_value_(
                            net.parameters(), clip_value=2.0
                        )

                        optimizer.step()

                        # 10 iterに1度、lossを表示
                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(
                                "イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec."
                                .format(iteration, loss.item(), duration)
                            )
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()

        t_epoch_finish = time.time()
        print('-----------')
        print(
            "epoch: {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Loss:{:.4f}"
            .format(epoch, epoch_train_loss, epoch_val_loss)
        )
        print(
            "timer: {:.4f} sec"
            .format(t_epoch_finish - t_epoch_start)
        )
        t_epoch_start = time.time()

        # ログの保存
        log_epoch = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # ネットワークを保存する
        if (epoch % 10) == 0:
            torch.save(net.state_dict(), "weights/ssd300_" +
                       str(epoch) + ".pth")


train_and_eval()
