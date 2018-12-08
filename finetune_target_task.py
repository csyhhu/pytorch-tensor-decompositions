"""
This code trains target task based on decomposed model (fine-tuned by source task or not).
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet, CIFARNet2
from utils.dataset import get_dataloader
from utils.train import validate, progress_bar
import itertools

from tensorboardX import SummaryWriter

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

    return 100.0 * correct / total

# ----------------------------------
use_cuda = torch.cuda.is_available()
select_type = 4
# Pretrained model
if select_type == 1:
    pretrain_path = './checkpoint/tucker_CIFARNet9_dual.pth'
elif select_type == 2:
    pretrain_path = './checkpoint/tucker_CIFARNet9_dual_2.pth'
elif select_type == 3:
    pretrain_path = './checkpoint/tucker_CIFARNet9_dual_2.pth'
elif select_type == 4:
    pretrain_path = './checkpoint/ft_tucker_CIFARNet9_dual_2.pth'
# Summary path
if select_type == 1:
    summary_path = './runs/Target-Finetune-Dual'
elif select_type == 2:
    summary_path = './runs/Target-Finetune-Dual-2'
elif select_type == 3:
    summary_path = './runs/Target-Finetune-Decomposed'
elif select_type == 4:
    summary_path = './runs/Target-Finetune-Finetune'

source_dataset_name = 'CIFAR9'
target_dataset_name = 'STL9'
n_epoch = 100
# ---------------------------------

net = CIFARNet() if select_type == 1 else CIFARNet2()
pretrain_param = torch.load(pretrain_path)
net.load_state_dict(pretrain_param)

if use_cuda:
    net.cuda()

train_laoder = get_dataloader(target_dataset_name, 'train', 128)
test_loader = get_dataloader(target_dataset_name, 'test', 128)
source_test_loader = get_dataloader(source_dataset_name, 'test', 128)

# print('\nFirst testing')
test(net, True, test_loader)
test(net, False, source_test_loader)
input('Does the result correct?')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# ---------------------------
# Select trainable parameters
# ---------------------------
trainable_parameters = list()
for named, param in net.named_parameters():

    if select_type == 1:
        if 'features_0_0' in named or 'features_0_2' in named or \
            'features_3_0' in named or 'features_3_2' in named or \
            'features_6_0' in named or 'features_6_2' in named:
            continue
    elif select_type == 2:
        if 'features_0_1' in named or \
            'features_3_1' in named or \
            'features_6_1' in named:
            continue

    trainable_parameters.append(param)

print('Trainable parameters: %d/%d' %(len(trainable_parameters), len(list(net.parameters()))))
optimizer = optim.SGD(trainable_parameters, lr=0.01, momentum=0.9, weight_decay=5e-4)
# ds
small_train_loss = 1e9
descend_count = 0
niter = 0
stop_flag = False
best_test_acc = 0

writer = SummaryWriter(summary_path)

for epoch in range(n_epoch):

    if stop_flag:
        break

    print('\nEpoch: %d' %epoch)

    net.train()

    total = 0
    train_loss = 0
    correct_t = 0
    correct_s = 0

    for batch_idx, (inputs, targets) in enumerate(train_laoder):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Training target task
        _, output = net(inputs, target_path=True)
        losses = nn.CrossEntropyLoss()(output, targets)
        train_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Get record
        _, predicted = torch.max(output.data, dim=1)
        correct_t += predicted.eq(targets).cpu().sum().item()
        total += targets.size(0)

        progress_bar(batch_idx, len(train_laoder),
                     "Loss: %.3f | Acc: %.3f%%" %
                     (100.0 * train_loss / (batch_idx + 1),
                      100.0 * correct_t / total))

        writer.add_scalar('Train/Loss', train_loss / (batch_idx + 1), niter)
        writer.add_scalar('Train/Accuracy', 100.0 * train_loss / (batch_idx + 1), niter)

        niter += 1

    if train_loss < small_train_loss:
        # if 100. * float(correct) / float(total) > best_train_acc:
        small_train_loss = train_loss
        best_train_acc = 100. * float(correct_t) / float(total)
        descend_count = 0

    else:
        descend_count += 1

    print('Training loss: %.3f, descend count: %d' % (train_loss, descend_count))

    if descend_count >= 3:

        descend_count = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            print('Learning rata: %e' % param_group['lr'])
            # When learning rate is too small, break
            if param_group['lr'] < 1e-10:
                stop_flag = True
                break

    # print('\nTest with source data')
    # test(net, False, test_loader)
    print('\nTest with test data')
    acc = test(net, True, test_loader)
    writer.add_scalar('Test/Accuracy', acc, niter)
    if best_test_acc < acc:
        print('Best test accuracy')
        best_test_acc = acc

print('Best test accuracy: %.3f' %best_test_acc)