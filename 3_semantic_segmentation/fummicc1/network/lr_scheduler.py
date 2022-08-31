import math

# lrのスケジューラ
def lambda_epoch(epoch: int):
    max_epoch = 30
    return math.pow((1 - epoch/max_epoch), 0.9)