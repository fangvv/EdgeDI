import torch
import torch.nn as nn
import psutil
# from memory_profiler import profile
import os
default_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class vgg13(nn.Module):
    def __init__(self, layer_nums = None):
        super(vgg13, self).__init__()
        if layer_nums is None:
            self.layer_nums = default_layer
        else:
            self.layer_nums = layer_nums

        self.inchannels = 3
        self.features = self._make_layer(self.layer_nums)
        self.linear = nn.Sequential(
            nn.Linear(6*6*self.layer_nums[-2], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )


    def _make_layer(self, layer_nums):
        layers = []
        for v in layer_nums:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                CONV2d = nn.Conv2d(self.inchannels, v, kernel_size=3, padding=1, bias=False)
                layers += [CONV2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                self.inchannels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        count = 0
        for module in self.features:
                x = module(x)
        x = x.view(x.size(0), -1)
        for module in self.linear:
            x = module(x)
        return x


if __name__ == "__main__":
    pruning_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    net = vgg13(pruning_layer)
    print(net)
    x = torch.rand([1, 3, 200, 200])
    y = net(x)
    print(y.shape)