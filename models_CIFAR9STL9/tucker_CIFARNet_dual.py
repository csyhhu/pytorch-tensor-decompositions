"""
A dual path CIFARNet for testing MTL/DA in compressed network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models_CIFAR10.VATNet import Flatten

class CIFARNet(nn.Module):

    def __init__(self):
        super(CIFARNet, self).__init__()
        '''
        self.features = nn.Sequential(

            nn.Conv2d(3, 2, 1, 1, bias=False),
            nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(8, 32, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),

            nn.Conv2d(32, 22, 1, 1, bias=False),
            nn.Conv2d(22, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(4, 32, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2),

            nn.Conv2d(32, 4, 1, 1, bias=False),
            nn.Conv2d(4, 39, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(39, 64, 1, 1),

            nn.ReLU(),
            nn.AvgPool2d(3, 2)
        )
        '''
        self.classifier = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

        self.features_0_0 = nn.Conv2d(3, 2, 1, 1, bias=False)
        self.features_0_1 = nn.Conv2d(2, 8, 5, stride=1, padding=2, bias=False)
        self.features_0_1_t = nn.Conv2d(2, 8, 5, stride=1, padding=2, bias=False)
        self.features_0_2 = nn.Conv2d(8, 32, 1, 1)

        self.features_3_0 = nn.Conv2d(32, 22, 1, 1, bias=False)
        self.features_3_1 = nn.Conv2d(22, 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_3_1_t = nn.Conv2d(22, 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_3_2 = nn.Conv2d(4, 32, 1, 1)

        self.features_6_0 = nn.Conv2d(32, 4, 1, 1, bias=False)
        self.features_6_1 = nn.Conv2d(4, 39, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_6_1_t = nn.Conv2d(4, 39, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_6_2 = nn.Conv2d(39, 64, 1, 1)

    def forward(self, x, target_path = False):

        x = self.features_0_0(x)
        if target_path:
            x = self.features_0_1_t(x)
        else:
            x = self.features_0_1(x)
        x = self.features_0_2(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(x)

        x = self.features_3_0(x)
        if target_path:
            x = self.features_3_1_t(x)
        else:
            x = self.features_3_1(x)
        x = self.features_3_2(x)

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2)

        x = self.features_6_0(x)
        if target_path:
            x = self.features_6_1_t(x)
        else:
            x = self.features_6_1(x)
        x = self.features_6_2(x)

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2)

        adapt = x.view(-1, 576)
        x = self.classifier(adapt)

        return adapt, x


class CIFARNet2(nn.Module):

    def __init__(self):
        super(CIFARNet2, self).__init__()

        self.features_0_0 = nn.Conv2d(3, 2, 1, 1, bias=False)
        self.features_0_0_t = nn.Conv2d(3, 2, 1, 1, bias=False)
        self.features_0_1 = nn.Conv2d(2, 8, 5, stride=1, padding=2, bias=False)
        self.features_0_2 = nn.Conv2d(8, 32, 1, 1)
        self.features_0_2_t = nn.Conv2d(8, 32, 1, 1)

        self.features_3_0 = nn.Conv2d(32, 22, 1, 1, bias=False)
        self.features_3_0_t = nn.Conv2d(32, 22, 1, 1, bias=False)
        self.features_3_1 = nn.Conv2d(22, 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_3_2 = nn.Conv2d(4, 32, 1, 1)
        self.features_3_2_t = nn.Conv2d(4, 32, 1, 1)

        self.features_6_0 = nn.Conv2d(32, 4, 1, 1, bias=False)
        self.features_6_0_t = nn.Conv2d(32, 4, 1, 1, bias=False)
        self.features_6_1 = nn.Conv2d(4, 39, kernel_size=5, stride=1, padding=2, bias=False)
        self.features_6_2 = nn.Conv2d(39, 64, 1, 1)
        self.features_6_2_t = nn.Conv2d(39, 64, 1, 1)

        self.classifier = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x, target_path = False):

        if target_path:
            x = self.features_0_0_t(x)
        else:
            x = self.features_0_0(x)

        x = self.features_0_1(x)

        if target_path:
            x = self.features_0_2_t(x)
        else:
            x = self.features_0_2(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(x)
        # ------------------------------------------------
        if target_path:
            x = self.features_3_0_t(x)
        else:
            x = self.features_3_0(x)

        x = self.features_3_1(x)

        if target_path:
            x = self.features_3_2_t(x)
        else:
            x = self.features_3_2(x)

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2)
        # ------------------------------------------------
        if target_path:
            x = self.features_6_0_t(x)
        else:
            x = self.features_6_0(x)

        x = self.features_6_1(x)

        if target_path:
            x = self.features_6_2_t(x)
        else:
            x = self.features_6_2(x)

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2)

        adapt = x.view(-1, 576)
        x = self.classifier(adapt)

        return adapt, x


if __name__ == '__main__':
    net = CIFARNet()
    inputs = torch.rand([10, 3, 32, 32])
    outputs = net(inputs)
    print(outputs.shape)