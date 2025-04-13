from .caltech101 import Caltech101
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch


dataset_list = {
                "caltech101": Caltech101,
                "imagenet-a": ImageNetA,
                "imagenet-v": ImageNetV2,
                "imagenet-r": ImageNetR,
                "imagenet-s": ImageNetSketch,
                }


def build_dataset(dataset, root_path):
    return dataset_list[dataset](root_path)