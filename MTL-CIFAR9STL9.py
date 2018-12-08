"""
An experimental code for verifying whether low-rank approximation of tensor
can leads to improvement of multi-task training:

A pretrained CIFAR9 model is decomposed by CP/Tucker, with an extra coefficient tensor is added
for STL9 finetune.
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet2 as CIFARNet
from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet
from utils.dataset import get_dataloader
from utils.train import validate, progress_bar
import itertools


def test(net, target_path, test_loader, use_cuda = True):

    correct = 0
    total = 0

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        _, output = net(inputs, target_path)
        _, predicted = torch.max(output, dim=1)
        correct += predicted.eq(targets).cpu().sum().item()
        total += targets.size(0)

        progress_bar(batch_idx, len(test_loader), "Acc: %.3f%%" %(100.0 * correct / total))

# ----------------------------------
use_cuda = torch.cuda.is_available()
pretrain_path = './checkpoint/ft_tucker_CIFARNet9_dual.pth'
source_dataset_name = 'CIFAR9'
target_dataset_name = 'STL9'
n_epoch = 10
# ---------------------------------

net = CIFARNet()
pretrain_param = torch.load(pretrain_path)
net.load_state_dict(pretrain_param)

if use_cuda:
    net.cuda()

source_train_loader = get_dataloader(source_dataset_name, 'train', 128)
source_test_loader = get_dataloader(source_dataset_name, 'test', 128)
target_train_loader = get_dataloader(target_dataset_name, 'train', 128)
target_test_loader = get_dataloader(target_dataset_name, 'test', 128)
n_source_ite = len(source_train_loader)
n_target_ite = len(target_train_loader)
n_ite = max(len(source_train_loader), len(target_train_loader))
if n_source_ite < n_target_ite:
    source_train_loader = itertools.cycle(source_train_loader)
else:
    target_train_loader = itertools.cycle(target_train_loader)

# print('\nFirst testing')
# test(net, True, target_test_loader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

for epoch in range(n_epoch):

    print('\nEpoch: %d' %epoch)

    net.train()

    total = 0
    correct_t = 0
    correct_s = 0

    for batch_idx, ((x_s, y_s), (x_t, y_t)) \
            in enumerate(zip(source_train_loader, target_train_loader)):

        if x_s.shape != x_t.shape:
            continue

        if batch_idx >= (n_ite-1): break

        if use_cuda:
            x_s, y_s, x_t, y_t = \
                x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()

        # Training source task
        _, logits_s = net(x_s)
        losses_s = nn.CrossEntropyLoss()(logits_s, y_s)

        # Training target task
        _, logits_t = net(x_t, target_path=True)
        losses_t = nn.CrossEntropyLoss()(logits_t, y_t)

        optimizer.zero_grad()
        (losses_t + losses_s).backward()
        optimizer.step()

        # Get record
        _, predicted = torch.max((logits_s).data, dim=1)
        correct_s += predicted.eq(y_s).cpu().sum().item()
        _, predicted = torch.max((logits_t).data, dim=1)
        correct_t += predicted.eq(y_t).cpu().sum().item()
        total += y_s.size(0)

        progress_bar(batch_idx, n_ite,
                     "Src acc: %.3f%% | Tgt acc: %.3f%%" %
                     (100.0 * correct_s / total,
                      100.0 * correct_t / total))

    print('\nTest with source data')
    test(net, False, source_test_loader)
    print('Test with target data')
    test(net, True, target_test_loader)