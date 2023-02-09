import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import metrics

import os
import argparse

from models import *
from utils import progress_bar
import visualization
from metrics import ECELoss

np.random.seed(0)


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
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


if not os.path.isdir('graphs'):
    os.mkdir('graphs')

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

def get_reliability(net, conf_thresh):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prob = F.softmax(outputs, dim=1)

            top_p, top_class = prob.topk(1, dim = 1)

            # if top_p> conf_thresh:
            greater_than_conf = (top_p>conf_thresh).float()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += greater_than_conf * predicted.eq(targets).sum().item()
            

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        accuracy = correct/total

    return accuracy


def run_visualizations(checkpoint_path = "./checkpoint/ckpt.pth", conf=0.5, alpha = 0.001):
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    # checkpoint = torch.load('./checkpoint/model_zeroed.pth')
    criterion = nn.CrossEntropyLoss()

    net = ResNet18()
    net = net.to(device)

    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['net'])

    correct = 0
    total = 0

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in testloader:
        # for images, labels in trainloader:
            #outputs are the the raw scores!
            
            logits = net(images.to(device))
            #add data to list
            logits_list.append(logits)
            labels_list.append(labels.to(device))
            #convert to probabilities
            output_probs = F.softmax(logits,dim=1)
            #get predictions from class
            probs, predicted = torch.max(output_probs, 1)
            #total
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

    # print('Accuracy of the network on the test images: %d %%' % (
    #     100.0 * correct / total))
    accuracy = 100.0 * correct / total
    # print(total)

    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()

    ############
    # Visualizations
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
    plt_test.savefig(f'graphs/conf_histogram_test_{conf}_{alpha}.png',bbox_inches='tight')
    #plt_test.show()

    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
    plt_test_2.savefig(f'graphs/rel_diagram_test_{conf}_{alpha}.png',bbox_inches='tight')

    # ECE 
    ece_calculator = ECELoss()
    ece_loss = ece_calculator.loss(logits_np, labels_np)
    # print ("ECE:", ece_loss)

    return accuracy, ece_loss

if __name__=="__main__":
    acc, ece_loss = run_visualizations('./checkpoint/ckpt.pth', conf=0, alpha=0)
    print (acc, ece_loss)