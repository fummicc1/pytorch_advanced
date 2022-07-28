from typing import List
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import numpy as np


class Anno_xml2list:
    def __init__(self, classes: List) -> None:
        self.classes = classes

    def __call__(self, xml_path: str, width: int, height: int) -> np.ndarray:
        """一枚の画像に対する「XML形式のアノテーションデータ」を画像サイズで規格化してからリスト形式に変換する
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納します
        ret = []
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体の数だけループする
        for obj in xml.iter("object"):
            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue

            # 一つの物体に対するアノテーションを格納するリスト
            bndbox = []

            # 物体名
            name = obj.find("name").text.lower().strip()
            # バウンディングボックスの情報
            bbox = obj.find("bndbox")

            # アノテーションの[xmin, ymin, xmax, ymax]を用いてbboxを規格化
            pts = ["xmin", "ymin", "xmax", "ymax"]

            for pt in pts:
                # アノテーションデータ（VOC）は原点が(1,1)なので-1をして原点を(0,0)に調整
                cur_pixel = int(bbox.find(pt).text) - 1

                if pt in ["xmin", "xmax"]:
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                bndbox.append(cur_pixel)

            # アノテーションクラスのindexを取得
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret)
