import torch
import torch.nn as nn
import time
import torch.nn.functional as F
#块层数，第一层一个，剩余为块内层数
def_layer = [64, 64, 64, 128, 128, 256, 256, 512, 512]

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, strid):
        super(BasicBlock, self).__init__()

        self.strid_ = strid
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=strid, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=strid, padding=0, bias=False),
            nn.BatchNorm2d(outchannel)
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.strid_ == 2:
            xsout = x[:, :, 3:x.shape[2]-3, 3:x.shape[3]-3]
        else:
            xsout = x[:, :, 2:x.shape[2]-2, 2:x.shape[3]-2]


        sout = self.shortcut(xsout)
        # sout = self.shortcut(x)
        out = out + sout
        # out = x
        # if self.strid_ != 2:
        #     out = self.conv1(x)
        # out = self.conv2(out)
        # if self.strid_ == 2:
        #     xsout = x[:, :, 1:x.shape[2]-1, 1:x.shape[3]-1]
        # else:
        #     xsout = x[:, :, 2:x.shape[2]-2, 2:x.shape[3]-2]
        # if self.strid_ != 2:
        #     sout = self.shortcut(xsout)
        # else:
        #     sout = xsout
        # out = out + sout
        out = F.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, layer_num=None):
        super(resnet18, self).__init__()

        if layer_num == None:
            self.layer_num = def_layer
        else:
            self.layer_num = layer_num

        self.feature = self._make_layer(BasicBlock)
        self.fc = nn.Linear(self.layer_num[-1]*3*3, 6)

    def forward(self, x):
        index = 0
        for module in self.feature:
            if 8 <= index < 9:
                x = module(x)
            index += 1
        # out = F.avg_pool2d(x, 2)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return x


    def _make_layer(self, BasicBlock):
        self.inchannel = 3
        strid = [2, 1, 2, 1, 2, 1, 2, 1]
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
    layer_name = [64, 64, 64, 128, 128, 256, 256, 512, 512]
    net = resnet18(layer_name)
    x = torch.rand([1, 3, 200, 200])
    print(net)