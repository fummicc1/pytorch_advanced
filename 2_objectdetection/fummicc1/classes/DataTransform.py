from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans


class DataTransform:
    def __init__(self):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
            ]),
            "test": Compose([
                
            ])
        }
        