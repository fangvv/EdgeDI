import torch
import torch.nn as nn
import os
# from memory_profiler import profile
import psutil
default_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class Feature_enhancement(nn.Module):
    def __init__(self, channel):
        super(Feature_enhancement, self).__init__()
        self.globalAvgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.globalAvgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out*x
        return out


class vgg13Block(nn.Module):
    def __init__(self, layer_nums = None):
        super(vgg13Block, self).__init__()
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

                outchannel = int(v/4)
                Conv2d_down = nn.Conv2d(self.inchannels, outchannel, kernel_size=1, padding=0, bias=False)
                Conv2d_conv = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, bias=False)
                Conv2d_up = nn.Conv2d(outchannel, v, kernel_size=1, padding=0, bias=False)
                layers += [Conv2d_down, Conv2d_conv, Feature_enhancement(outchannel), Conv2d_up, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]

                self.inchannels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        for module in self.features:
            x = module(x)
        x = x.view(x.size(0), -1)
        for module in self.linear:
            x = module(x)
        return x


if __name__ == "__main__":
    net = vgg13Block()
    x = torch.rand([1, 3, 200, 200])
    y = net(x)
    print(y.shape)