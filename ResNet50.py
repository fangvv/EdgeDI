import torch
import torch.nn as nn
import torch.nn.functional as F
#块层数，第一层一个，剩余为块内层数
def_layer = [64, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, strid):
        super(BasicBlock, self).__init__()

        temp_out_channel = int(outchannel/4)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, temp_out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_out_channel, temp_out_channel, kernel_size=3, stride=strid, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_out_channel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=strid, padding=0, bias=False),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.conv1(x)

        xsout = x[:, :, 1:x.shape[2]-1, 1:x.shape[3]-1]

        s_out = self.shortcut(xsout)

        out = out + s_out
        out = F.relu(out)
        return out


class resnet50(nn.Module):
    def __init__(self, layer_num=None):
        super(resnet50, self).__init__()

        if layer_num == None:
            self.layer_num = def_layer
        else:
            self.layer_num = layer_num

        self.feature = self._make_layer(BasicBlock)
        self.fc = nn.Linear(self.layer_num[-1]*3*3, 6)

    def forward(self, x):
        index = 0
        for module in self.feature:
            if 14 <= index < 17:
                x = module(x)
            index += 1
        # out = F.avg_pool2d(x, 2)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return x


    def _make_layer(self, BasicBlock):
        self.inchannel = 3
        strid = [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        layer = []
        for i in range(len(self.layer_num)):
            outchannel = self.layer_num[i]
            if i == 0:
                temp_layer = nn.Sequential(
                    nn.Conv2d(self.inchannel, outchannel, kernel_size=3, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(self.layer_num[i]),
                    nn.ReLU(inplace=True)
                )
            else:
                temp_layer = BasicBlock(self.inchannel, outchannel, strid[i-1])
            layer.append(temp_layer)
            self.inchannel = outchannel
        return nn.Sequential(*layer)


if __name__ == "__main__":
    layer_name = [64, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
    net = resnet50(layer_name)
    x = torch.rand([1, 3, 200, 200])
    print(net)