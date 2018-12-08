"""
A code to fine-tune low-rank
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models_CIFAR9STL9.CIFARNet import CIFARNet

from utils.dataset import get_dataloader
from utils.train import progress_bar, train, validate

# --------------------------
use_cp = False
decomposed_path = './checkpoint/%s_CIFARNet9.p' %('cp' if use_cp else 'tucker')
use_cuda = torch.cuda.is_available()
n_epoch = 10
# --------------------------

# net = CIFARNet()
net = torch.load(decomposed_path)
train_loader = get_dataloader('CIFAR9', 'train', 128)
test_loader = get_dataloader('CIFAR9', 'test', 128)
target_test_loader = get_dataloader('STL9', 'test', 128)
if use_cuda:
    net.cuda()

print('First validation')
validate(net, test_loader)

optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.01)
train(net, train_loader, optimizer, n_epoch=n_epoch, val_loader=test_loader)
# for epoch in range(n_epoch):
torch.save(net, './checkpoint/ft_%s_CIFARNet9.p' %('cp' if use_cp else 'tucker'))
print('Target test')
validate(net, target_test_loader)