import torch
import torch.nn as nn
# from models_CIFAR10.VATNet import Flatten

class CIFARNet(nn.Module):

    def __init__(self):
        super(CIFARNet, self).__init__()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 2, 1, 1, bias=False),
                nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Conv2d(8, 32, 1, 1),
            ),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),

            nn.Sequential(
                nn.Conv2d(32, 22, 1, 1, bias=False),
                nn.Conv2d(22, 4, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Conv2d(4, 32, 1, 1),
            ),
            nn.ReLU(),
            nn.AvgPool2d(3, 2),

            nn.Sequential(
                nn.Conv2d(32, 4, 1, 1, bias=False),
                nn.Conv2d(4, 39, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Conv2d(39, 64, 1, 1),
            ),

            nn.ReLU(),
            nn.AvgPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x, check_mode = False):

        # if not check_mode:
        x = self.features(x)
        x = x.view(-1, 576)
        x = self.classifier(x)
        return x
        # else:
        #     for layer in self.features:
        #         x = layer(x)
        #         print(type(layer))
        #         print(x.shape)
        #     x = x.view(-1, 576)
        #     for layer in self.classifier:
        #         x = layer(x)
        #         print(x.shape)
        #     return x


if __name__ == '__main__':


    from utils.dataset import get_dataloader
    from utils.train import validate

    net = CIFARNet()
    # inputs = torch.rand([10, 3, 32, 32])
    # net(inputs, check_mode=True)
    decomposed_net = torch.load('./checkpoint/tucker_CIFARNet9.p')
    torch.save(decomposed_net.state_dict(), './checkpoint/tucker_CIFARNet9.pth')
    '''
    net.load_state_dict(decomposed_net.state_dict())

    net.cuda()
    decomposed_net.cuda()
    test_loader = get_dataloader('CIFAR9', 'test', 128)
    validate(net, test_loader)
    validate(decomposed_net, test_loader)
    '''
    # net = CIFARNet()
    # for named, param in net.named_parameters():
    #     print(named)