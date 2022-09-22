#测试卷积运行时间
import torch
import os
import numpy as np
from memory_profiler import profile
from ResNet50 import resnet50
from VGG16Block import vgg16Block

pruning_layer = [64, 63, 'M', 128, 128, 'M', 254, 253, 255, 'M', 507, 512, 511, 'M', 512, 512, 512, 'M']
x = torch.rand([1, 3, 200, 200])
model = vgg16Block(pruning_layer)


def _save_model():
    list_name = _get_name()
    path = "./SaveInfo/vgg13/R4/"
    if not os.path.exists(path):
        os.makedirs(path)

    for name in list_name:
        temp_np = model.state_dict()[name].cpu().numpy()
        np.save(path+"%s.ndim" % (name), temp_np)

    print("model saved in {}".format(path))


def _get_name():
    list_name = []
    for name in model.state_dict():
        list_name.append(name)
    return list_name


@profile
def test():
    y = model(x)
    print(y)


if __name__ == "__main__":
    test()
    # _save_model()