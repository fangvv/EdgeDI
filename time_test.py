#测试卷积运行时间
import time
import torch
import gc
from VGG13Block import vgg13Block
from TTModel import TTNet

device = torch.device('cpu')
pruning_layer = [62, 64, 'M', 128, 127, 'M', 256, 255, 'M', 508, 510, 'M', 508, 512, 'M']
model = vgg13Block(pruning_layer)
# model = TTNet()

x = torch.rand([1, 510, 6, 16])
time_list = []


def test():
    start = time.time()
    y = model(x)
    print(y.shape)
    end = time.time()
    print(end - start)
    time_list.append(end - start)
    gc.collect()


if __name__ == "__main__":
    for i in range(20):
        test()
    print("Max_time: %s"%(max(time_list)))
    print("Min_time: %s"%(min(time_list)))
    all_time = 0
    for i in range(20):
        all_time += time_list[i]
    print("Avg_time: %s" %((all_time-max(time_list)-min(time_list))/18))