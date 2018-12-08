"""
This code used the factorized/decomposed model, train target data by fixing certain layers
to verify the effectiveness of core layer in domain transfer
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models_CIFAR9STL9.tucker_CIFARNet import CIFARNet
from utils.dataset import get_dataloader
from utils.train import validate, progress_bar

# ----------------------------------
use_cuda = torch.cuda.is_available()
pretrain_path = './checkpoint/tucker_CIFARNet9.pth'
source_dataset_name = 'CIFAR9'
target_dataset_name = 'STL9'
n_epoch = 10
# ---------------------------------

net = CIFARNet()
pretrain_param = torch.load(pretrain_path)
net.load_state_dict(pretrain_param)

if use_cuda:
    net.cuda()

# source_train_loader = get_dataloader(source_dataset_name, 'train', 128)
# source_test_loader = get_dataloader(source_dataset_name, 'test', 128)
target_train_loader = get_dataloader(target_dataset_name, 'train', 128)
target_test_loader = get_dataloader(target_dataset_name, 'test', 128)

# validate(net, target_test_loader)

# Select trainable parameters
trainable_parameters = list()
for named, param in net.named_parameters():
    if 'features.0.1.weight' in named or 'features.3.1.weight' in named or 'features.6.1.weight' in named:
        continue
    else:
        trainable_parameters.append(param)

optimizer = optim.Adam(trainable_parameters, lr=1e-3)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(n_epoch):

    net.train()
    loss = 0
    correct = 0
    total = 0

    print('\n[Epoch: %d] \nTraining' %(epoch))

    for batch_idx, (inputs, targets) in enumerate(target_train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        losses = nn.CrossEntropyLoss()(outputs, targets)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss += losses.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        total += targets.size(0)

        progress_bar(batch_idx, len(target_train_loader), "Loss: %.3f | Acc: %.3f%%"
                     %(loss / (batch_idx + 1), 100.0 * correct / total))
    print('Test')
    validate(net, target_test_loader)