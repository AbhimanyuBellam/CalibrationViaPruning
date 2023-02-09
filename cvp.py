import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys

import os
import argparse

from models import *
from utils import progress_bar
import copy
import shutil

np.random.seed(0)

parser = argparse.ArgumentParser(description='CVP')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

# parser.add_argument('--epochs', type=int, help='num epochs')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()


def predict(inputs, targets, optimizer, net, wt_zeroed_correct):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)

    _, predicted = outputs.max(1)
    # print (predicted, targets)
    print ("output corretness? ", predicted.eq(targets).sum().item(), "/", targets.size(0))
    
    wt_zeroed_correct += predicted.eq(targets).sum().item()
    return wt_zeroed_correct


def get_low_conf_images(low_conf_thresh = 0.5):
    if not os.path.isdir('low_conf_data'):
        os.mkdir('low_conf_data')
        os.mkdir('low_conf_data/images')
        os.mkdir('low_conf_data/targets')
    
    elif len(os.listdir("low_conf_data/images"))>0:
        print ("Removing existing files")
        shutil.rmtree("low_conf_data/images")
        shutil.rmtree("low_conf_data/targets")
        os.mkdir('low_conf_data/images')
        os.mkdir('low_conf_data/targets')

    net = ResNet18()
    net = net.to(device)

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # net.train()
    net.eval()
    
    low_conf_inputs, low_conf_targets = [], []
    total =0 ; correct = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        # INFO: search for low conf images
        prob = F.softmax(outputs, dim=1)

        # print ("TOPK:",prob.topk(10, dim = 1) )
        top_p, top_class = prob.topk(1, dim = 1)
        # print ("Top prob:", top_p)
        confs_batch = (top_p<low_conf_thresh).float()
        # print ("confs_batch:",confs_batch)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print ("Correct:", correct )

        # is_correct = predicted.eq(targets).sum().item()
        is_correct = predicted.eq(targets)

        # INFO: Gather all low conf images:
        # Batch:
        with torch.no_grad():
            for i in range(len(confs_batch)):
                if confs_batch[i][0] ==1 and is_correct[i]:
                    low_conf_inputs.append(torch.tensor(inputs[i]).cpu().clone().detach())
                    low_conf_targets.append(torch.tensor(targets[i]).cpu().clone().detach())
                    torch.save(inputs[i],f"low_conf_data/images/img_{batch_idx}_{i}.pt")
                    torch.save(targets[i],f"low_conf_data/targets/target_{batch_idx}_{i}.pt")

        # One image at a time:
        # if confs_batch[0][0] ==1 and is_correct:
        #     low_conf_inputs.append(inputs)
        #     low_conf_targets.append(targets)
        #     torch.save(inputs[0],f"low_conf_data/images/img_{batch_idx}.pt")
        #     torch.save(targets[0],f"low_conf_data/targets/target_{batch_idx}.pt")
        
        progress_bar(batch_idx, len(testloader))

        
    
    return low_conf_inputs, low_conf_targets
        

def identify_weights(low_conf_images=None, low_conf_targets=None, epoch=1, conf=0.5, alpha = 0.001, batch_size = 100):    
    correct = 0
    total = 0

    path = "low_conf_data"
    images_temp = os.listdir(f"{path}/images")
    targets_temp = os.listdir(f"{path}/targets")

    d,w,h = torch.load(f"{path}/images/{images_temp[0]}").shape 
    print ("img size:",d,w,h)

    inputs = torch.Tensor(len(images_temp),d,w,h)
    # targets = torch.Tensor(len(targets_temp))
    targets= []

    for i in range(len(images_temp)):
        inputs[i] = torch.load(f"{path}/images/{images_temp[i]}")
        targets.append( torch.load(f"{path}/targets/{targets_temp[i]}"))
    
    targets = torch.tensor(targets, dtype = torch.long)

    # print(targets)

    # INFO: For 1 image at a time, batch_size =1
    wt_zeroed_correct =0

    # for batch_idx, (inputs, targets) in enumerate(low_conf_images):
    net = ResNet18()
    net = net.to(device)

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    net.train()
    # net.eval()

    initial_fc_weights =  copy.deepcopy(net.linear.weight.data)

    # net_temp_params = copy.deepcopy(net.parameters())

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)

    # INFO: search for low conf images
    prob = F.softmax(outputs, dim=1)

    # INFO: Weight update
    loss = criterion(outputs, targets)
    print ("Loss:",loss)
    loss.backward()
    optimizer.step()
    
    # INFO: weight change
    new_fc_weights = net.linear.weight.data

    delta_w = new_fc_weights - initial_fc_weights

    print ("Max Weight change:",delta_w.max()) 
    # break

    # INFO: set weights whose change is high for low conf images to 0 # gather indexes 
    bad_weight_indexes = []
    weights_zeroed_current = []
    for i, row in enumerate(delta_w):
        for j, value in enumerate(row):
            if value>alpha:
                bad_weight_indexes.append([i,j])
                # new_fc_weights[i][j] = 0
                initial_fc_weights[i][j] = 0
                # weights_zeroed_current.append(new_fc_weights[i][j])
                weights_zeroed_current.append(value)
    
    # set current weights to the initial weights where some have been zeroed.
    net.linear.weight.data = initial_fc_weights
    
    num_zeroed_weights = len(bad_weight_indexes)
    print (f"Making {len(bad_weight_indexes)} weights zeroes")

    # INFO: predict again
    wt_zeroed_correct = predict(inputs, targets, optimizer, net, wt_zeroed_correct)

    # print ("____________")
    print ("after zeroed correct:", wt_zeroed_correct, "total:",len(inputs))

    # saving model
    print('Saving..')
    checkpoint_save_path = f'./checkpoint/model_zeroed_{conf}_{alpha}.pth'
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, checkpoint_save_path)

    return checkpoint_save_path, num_zeroed_weights


if __name__=="__main__":

    # print ("Num low_conf = ", len(get_low_conf_images()[0]))

    low_conf_images, low_conf_targets =get_low_conf_images(low_conf_thresh = 0.4)

    print ("Num images: ", len(low_conf_images))
    print (len(low_conf_targets))
    print (low_conf_targets)

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

ALSO, try:

1. Create batch of low conf images instead of doing on single images. 

2. get the distribution of the FC layer weights?
"""


