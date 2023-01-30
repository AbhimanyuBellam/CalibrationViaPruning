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

np.random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.isdir('graphs'):
    os.mkdir('graphs')

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

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


def rel_diagram_sub(accs, confs, ax, M = 10, name = "Reliability Diagram", xname = "Confidence", yname="Accuracy"):

    acc_conf = np.column_stack([accs,confs])
    acc_conf.sort(axis=1)
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1/M
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2)

    # Next add error lines
    #for i in range(M):
        #plt.plot([i/M,1], [0, (M-i)/M], color = "red", alpha=0.5, zorder=1)

    #Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "blue", label="Outputs", zorder = 3)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0,1], [0,1], linestyle = "--")
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(name, fontsize=24)
    ax.set_xlabel(xname, fontsize=22, color = "black")
    ax.set_ylabel(yname, fontsize=22, color = "black")


correct = 0
total = 0

# First: collect all the logits and labels for the validation set
logits_list = []
labels_list = []

with torch.no_grad():
    for images, labels in testloader:
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

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print(total)

logits_np = logits.cpu().numpy()
labels_np = labels.cpu().numpy()

############
#visualizations
conf_hist = visualization.ConfidenceHistogram()
plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
plt_test.savefig('graphs/conf_histogram_test.png',bbox_inches='tight')
#plt_test.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
plt_test_2.savefig('graphs/rel_diagram_test.png',bbox_inches='tight')