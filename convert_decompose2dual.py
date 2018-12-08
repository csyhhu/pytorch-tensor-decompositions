"""
This code convert weights in decomposed (fine-tune or not) into dual path corresponding models
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet2 as CIFARNet
# from models_CIFAR9STL9.tucker_CIFARNet_dual import CIFARNet
# from models_CIFAR9STL9.tucker_CIFARNet import CIFARNet
from utils.dataset import get_dataloader
from utils.train import validate
from utils.miscellaneous import progress_bar

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

# decompose_net = torch.load('./checkpoint/ft_tucker_CIFARNet9.p')
decompose_net = torch.load('./checkpoint/tucker_CIFARNet9.p')
net = CIFARNet()

d_state_dict = decompose_net.state_dict()
state_dict = net.state_dict()

for name, param in decompose_net.state_dict().items():
    # Analysis name
    if name in state_dict:
        state_dict[name] = param
    else:
        split_name = name.split('.')
        new_name = split_name[0] + '_' + split_name[1] + '_' + split_name[2] + '.' + split_name[3]
        if new_name in state_dict:
            state_dict[new_name] = param
        else:
            print('Param not found in state_dict: %s' %new_name)

        new_name_t = split_name[0] + '_' + split_name[1] + '_' + split_name[2] + '_t.' + split_name[3]
        if new_name_t in state_dict:
            state_dict[new_name_t] = param

torch.save(state_dict, './checkpoint/tucker_CIFARNet9_dual_2.pth')
# torch.save(state_dict, './checkpoint/tucker_CIFARNet9_dual.pth')
# torch.save(state_dict, './checkpoint/tucker_CIFARNet9.pth')

# For verification
net.load_state_dict(state_dict)
net.cuda()
decompose_net.cuda()
source_test_loader = get_dataloader('CIFAR9', 'test', 100)
target_test_loader = get_dataloader('STL9', 'test', 100)
print('Test with target dataset')
test(net, True, target_test_loader)
print('Test with source dataset')
test(net, False, source_test_loader)
# test(decompose_net, test_loader)