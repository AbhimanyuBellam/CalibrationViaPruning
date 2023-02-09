import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import pandas as pd

import os
import argparse
from collections import Counter

sys.path.append('../CalibrationViaPruning')
# from models import *
from utils import progress_bar
import copy
import shutil
import matplotlib.pyplot as plt

from cvp import get_low_conf_images, identify_weights
from conf_histogram import run_visualizations

def ece_vs_conf_and_alpha():
    confidences =[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # alphas = [0.003, 0.003, 0.0025, 0.0025, 0.002, 0.002, 0.0015, 0.001, 0.001, 0.0005, 0.0005 ]
    # start = 0.002s
    # alphas = [ start*(1 - (i+1/(len(confidences)))) for i in range(len(confidences)) ]
    alphas = [0.001 for i in range(11)]
    print (confidences, alphas)
    num_low_conf_images = []

    assert len(confidences)==len(alphas)

    num_zeroed_weights_list = []
    ece_values = []
    accuracies = []
    low_conf_classes_dicts_list = []

    for i in range(len(confidences)):
        print ("____\nConf:", confidences[i])
    
        low_conf_images, low_conf_targets = get_low_conf_images(confidences[i])
        # print (low_conf_targets)
        low_conf_classes_dict = {x.item():low_conf_targets.count(x.item()) for x in low_conf_targets}
        low_conf_classes_dict = dict(sorted(low_conf_classes_dict.items()))
        print ("Num images: ", len(low_conf_images))
        print ("Num target: ", len(low_conf_targets))
        checkpoint_path, num_zeroed_weights = identify_weights(conf=confidences[i], alpha=alphas[i])

        accuracy, ece_loss = run_visualizations(checkpoint_path=checkpoint_path, conf=confidences[i], alpha=alphas[i])

        print("Acc:",accuracy, "ECE:",  ece_loss)
        num_zeroed_weights_list.append(num_zeroed_weights)
        num_low_conf_images.append(len(low_conf_images))
        ece_values.append(ece_loss)
        accuracies.append(accuracy)
        low_conf_classes_dicts_list.append(low_conf_classes_dict)
        print(low_conf_classes_dict)
    results_df = pd.DataFrame(list(zip(confidences, num_low_conf_images, alphas, num_zeroed_weights_list, ece_values, accuracies, low_conf_classes_dicts_list))
                            , columns = ["confidence", "Num of low conf images", "alpha", "Number of weights zeroed", "ECE", "Accuracy","Classes present"])

    # plt.plot()
    print (low_conf_classes_dicts_list)
    results_df.to_csv("results/ece_vs_conf_and_alpha.csv")


if __name__=="__main__":
    ece_vs_conf_and_alpha()



    