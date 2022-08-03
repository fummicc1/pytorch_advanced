import pandas as pd
from classes.vgg import make_vgg
from classes.extras import make_extras
from classes.loc_conf import make_loc_conf
from classes.DBox import DBox
from classes.SSD import SSD

# vggの動作確認
vgg_test = make_vgg()
print("vgg_test", vgg_test)

# extrasの動作確認
extras_test = make_extras()
print("extras_test", extras_test)

# loc_confの動作確認
loc_test, conf_test = make_loc_conf()
print(loc_test)
print(conf_test)

# DBoxの動作確認
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

# DBox作成
dbox = DBox(ssd_config)
dbox_list = dbox.make_dbox_list()

# DBoxの出力を確認する
print(pd.DataFrame(dbox_list.numpy()))

# SSDの動作確認
ssd_test = SSD(phase="train", cfg=ssd_config)
print(ssd_test)
