import torch.nn as nn
import torch.nn.functional as F

class PSPLoss(nn.Module):
    def __init__(self, aux_weight: int = 0.4):
        super().__init__()
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs (_type_): タプルになっている. outputs[0]がdecoderの出力, outputs[1]がauxの出力
            targets (_type_): _description_
        """
        loss = F.cross_entropy(outputs[0], targets, reduction="mean")
        loss_aux = F.cross_entropy(outputs[1], targets, reduction="mean")        
        return loss + self.aux_weight * loss_aux