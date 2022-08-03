from torch.autograd import Function
import torch.nn as nn
import torch

from classes.decode import decode


class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を実行する。

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            オフセット情報。
        conf_data: [batch_num, 8732,num_classes]
            検出の確信度。
        dbox_list: [8732,4]
            DBoxの情報

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、クラス、confのtop200、BBoxの情報）
        """

        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_data = self.softmax(conf_data)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):

            decoded_boxes = decode(loc_data[i], dbox_list)

            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):

                c_mask = conf_scores[cl].gt(self.conf_thresh)

                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output
