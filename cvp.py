import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import copy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

# parser.add_argument('--epochs', type=int, help='num epochs')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

with torch.no_grad():
    print(net.linear.weight.data)
    print(net.linear.weight.data.shape)

# print (net)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)


def identify_weights(epoch=1):
    print('\nEpoch: %d' % epoch)
    net.train()
    checker_loss = 0
    correct = 0
    total = 0

    # initial weights: 
    initial_fc_weights =  copy.deepcopy(net.linear.weight.data)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        print (loss)
        loss.backward()
        optimizer.step()

        new_fc_weights = net.linear.weight.data

        delta_w = new_fc_weights - initial_fc_weights

        print (delta_w.sum().item()) 
        # break

        # search for low conf images

        # checker_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print ("Correct:", correct)

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #  % (checker_loss/(batch_idx+1), 100.*correct/total, correct, total))


identify_weights()


"""
Q. Should I free layers to identify?
# # freeze all 
# for param in net.parameters():
#     param.requires_grad = False

# # unfreezing FC layer
# net.linear.weight.requires_grad = True
# net.linear.bias.requires_grad = True


    # param.requires_grad = False
"""


