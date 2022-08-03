import torch
from itertools import product
from math import sqrt


class DBox:
    def __init__(self, cfg):
        self.image_size = cfg['input_size']

        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])
        self.steps = cfg['steps']

        self.min_sizes = cfg['min_sizes']

        self.max_sizes = cfg['max_sizes']

        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):

                f_k = self.image_size / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)

        output.clamp_(max=1, min=0)

        return output
