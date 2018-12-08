"""
This code validate the *.pth file when generated
"""

import torch
from utils.dataset import get_dataloader
from utils.train import validate
from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet
# from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet2 as CIFARNet
# Initial model
net = CIFARNet()

pretrain_param = torch.load('./checkpoint/tucker_CIFARNet9_dual.pth')
# pretrain_param = torch.load('./checkpoint/tucker_CIFARNet9_dual_2.pth')
net.load_state_dict(pretrain_param)

# Load dataset
test_loader = get_dataloader('STL9', 'test', 100)

net.cuda()
validate(net, test_loader)