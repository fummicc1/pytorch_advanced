from typing import Optional
from torch.utils.data import DataLoader
import torch


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        # sample[0]は画像データ
        imgs.append(sample[0])
        # sample[1]はアノテーションのGT
        targets.append(torch.FloatTensor(sample[1]))
    # imgsはlist。サイズは[3, 300, 300]
    # torch.Tensorに変換し、[batch_size, 3, 300, 300]に変換
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです
    # リストの要素のサイズは[n, 5]
    # nは検出された物体の数

    return imgs, targets


class VOCDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: Optional[int] = 1, shuffle: bool = False, sampler=None, batch_sampler=None, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, od_collate_fn, pin_memory, drop_last, timeout,
                         worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
