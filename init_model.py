import torch
import torch.nn as nn
import argparse
from torchstat import stat

parser = argparse.ArgumentParser(description="Demo of argparse")

parser.add_argument('--fm', type=str, default='200-200', help="feature_map_size")
parser.add_argument('--mt', type=str, default='3-64-64', help="model_type")
parser.add_argument('--st', type=str, default='1-1', help="stride_type")
args = parser.parse_args()

class init_model(nn.Module):
    def __init__(self, model_type = None, stride_type = None):
        super(init_model, self).__init__()
        self.model_type = model_type
        self.stride_type = stride_type
        self.model = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(len(self.model_type)-1):
            CONV2d = nn.Conv2d(self.model_type[i], self.model_type[i+1], kernel_size=3,
                               padding=0, stride=self.stride_type[i],
                               bias=False)
            layers += [CONV2d, nn.ReLU(inplace=True), nn.BatchNorm2d(self.model_type[i+1])]

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    features = list(map(int, args.fm.split('-')))
    model_type = list(map(int, args.mt.split('-')))
    stride_type = list(map(int, args.st.split('-')))

    model = init_model(model_type=model_type, stride_type=stride_type)
    stat(model, [model_type[0], features[0], features[1]])