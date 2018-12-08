import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from torchvision.models import vgg16_bn
from models_CIFAR10.CIFARNet import CIFARNet
from utils.dataset import get_dataloader
from utils.miscellaneous import progress_bar

class Trainer:
    def __init__(self, train_path, test_path, model, optimizer):
        # self.train_data_loader = dataset.loader(train_path)
        # self.test_data_loader = dataset.test_loader(test_path)

        self.train_data_loader = get_dataloader('CIFAR10', "train", 128)
        self.test_data_loader = get_dataloader('CIFAR10', "test", 128)

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

    def test(self):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        print("Average prediction time", float(total_time) / (i + 1), i + 1)

        self.model.train()

    def train(self, epoches=10):
        # for i in range(epoches):
        #     print("Epoch: ", i)
        #     self.train_epoch()
        #     self.test()
        # print("Finished fine tuning.")
        for epoch in range(epoches):
            total = 0
            correct = 0
            loss = 0
            print('\nEpoch: %d' %epoch)
            for batch_idx, (inputs, targets) in enumerate(self.train_data_loader):

                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                losses = nn.CrossEntropyLoss()(outputs, targets)

                self.optimizer.zero_grad()
                losses.backward()
                loss += losses.item()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                progress_bar(batch_idx, len(self.train_data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_idx + 1),
                            100. * float(correct) / float(total), correct, total))
            self.test()

    def train_batch(self, batch, label):
        self.model.zero_grad()
        input = Variable(batch)
        self.criterion(self.model(input), Variable(label)).backward()
        self.optimizer.step()

    def train_epoch(self):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(batch.cuda(), label.cuda())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')

    if args.train:
        # model = ModifiedVGG16Model().cuda()
        model = CIFARNet().cuda()
        # optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.train(epoches = 10)
        torch.save(model, "./checkpoint/CIFARNet.p")

    elif args.decompose:
        # model = torch.load("./vgg16_bn-6c64b313.pth").cuda()
        # model = ModifiedVGG16Model().cuda()
        # model = vgg16_bn()
        # model.load_state_dict(torch.load("./vgg16_bn-6c64b313.pth"))
        model = torch.load('./checkpoint/CIFARNet.p')
        # model.load_state_dict(torch.load("./CIFARNet.pth"))
        model.eval()
        model.cpu()
        N = len(model.features._modules.keys())
        for i, key in enumerate(model.features._modules.keys()):

            if i >= N - 2:
                break
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape)//3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)

                model.features._modules[key] = decomposed

        torch.save(model, './checkpoint/cp_CIFARNet.p')


    elif args.fine_tune:
        # base_model = torch.load("./checkpoint/decomposed_CIFAR10.p")
        # model = torch.nn.DataParallel(base_model)
        model = torch.load("./checkpoint/cp_CIFARNet.p")

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()        

        if args.cp:
            # optimizer = optim.SGD(model.parameters(), lr=0.000001)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            # optimizer = optim.SGD(model.parameters(), lr=0.001)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)


        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.test()
        model.cuda()
        model.train()
        trainer.train(epoches=10)
        model.eval()
        trainer.test()