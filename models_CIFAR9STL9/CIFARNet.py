import torch
import torch.nn as nn
# from models_CIFAR10.VATNet import Flatten

class CIFARNet(nn.Module):

    def __init__(self):
        super(CIFARNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),

            nn.AvgPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),

            nn.AvgPool2d(3, 2),
            # Flatten(576),

        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 576)
        x = self.classifier(x)
        return x
        # for layer in self.features:
        #     x = layer(x)
        #     print(x.shape)
        # x = x.view(-1, 576)
        # for layer in self.classifier:
        #     x = layer(x)
        #     print(x.shape)
        # return x


if __name__ == '__main__':

    net = CIFARNet()
    inputs = torch.rand([10, 3, 32, 32])
    # inputs.requires_grad_()
    # targets = torch.rand([10, 9])
    outputs = net(inputs)
    # losses = nn.MSELoss()(outputs, targets)
    # losses.backward()
    # print(inputs.grad[0,0,0:5,0:5])