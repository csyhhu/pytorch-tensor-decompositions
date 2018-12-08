"""
Codes to train source/target-only model with CIFAR9 and STL9
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

from utils.dataset import get_dataloader
from utils.train import progress_bar, train, validate
# from models_CIFAR9STL9 import convnet
# from models_CIFAR9STL9.resnet import resnet20_cifar
from models_CIFAR9STL9.tucker_CIFARNet import CIFARNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# ---------------------------------------------------------------
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# model_name = 'ResNet20'
# model_name = 'convnet'
model_name = 'tucker_CIFARNet'
source_dataset = 'CIFAR9'
target_dataset = 'STL9'
# source_dataset = 'STL9'
# target_dataset = 'CIFAR9'
# ----------------------------------------------------------------

source_train_loader = get_dataloader(source_dataset, 'train', 128)
source_test_loader = get_dataloader(source_dataset, 'test', 128)
target_test_loader = get_dataloader(target_dataset, 'test', 128)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s_%s_ckpt.t7' % (model_name, source_dataset))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    net = CIFARNet()
    net.load_state_dict(torch.load('ft'))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

break_count = 0
for epoch in range(start_epoch, start_epoch+200):
    print('\nEpoch: %d' %epoch)
    train(net, source_train_loader, optimizer=optimizer, n_epoch=1)
    acc_s = validate(net, source_test_loader)
    acc_t = validate(net, target_test_loader)

    if acc_s > best_acc:
        print('Saving..')
        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')

        state = {
            'net': net.module if use_cuda else net,
            'acc': acc_s,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s_%s_ckpt.t7' % (model_name, source_dataset))
        best_acc = acc_s
        torch.save(net.module.state_dict() if use_cuda else net.state_dict(),
                   './%s_%s.pth' %(model_name, source_dataset))
        break_count = 0

    else:
        break_count += 1

    if break_count >= 5:
        break
