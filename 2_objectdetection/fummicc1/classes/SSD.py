import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def forward(self, x):
        sources = list()  
        loc = list()  
        conf = list()  

        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)
        
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())        
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":              
            return self.detect(output[0], output[1], output[2])
        else:  
            return output
            