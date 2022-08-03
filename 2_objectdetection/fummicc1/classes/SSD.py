import torch.nn as nn
from classes.vgg import make_vgg
from classes.extras import make_extras
from classes.l2norm import L2Norm
from classes.loc_conf import make_loc_conf
from classes.DBox import DBox
from classes.Detect import Detect


class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg["num_classes"]

        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == 'inference':
            self.detect = Detect()
