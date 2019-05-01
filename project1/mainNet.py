# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:21:22 2019

@author: Charles
"""

import dlc_practical_prologue as prologue

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt

train_input, train_target, train_classes, \
    test_input, test_target, test_classes = prologue.generate_pair_sets(1000)


# Network
class Classifier_Net(nn.Module):
    def __init__(self):
        super(Classifier_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*3*3, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(-1, 64*3*3)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Comparison_Net(nn.Module):
    def __init__(self):
        super(Comparison_Net, self).__init__()
        self.fc1 = nn.Linear()


# Training function
def train_model(model, train_input, train_classes, mini_batch_size):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-2
    for e in range(30):
        sum_loss = 0
        for i in range(2):
            for b in range(0, train_input.size(0), mini_batch_size):
                output = model(train_input.narrow(1,i,1).narrow(0, b, mini_batch_size))
                loss = criterion(output, train_classes.narrow(1,i,1).narrow(0, b, mini_batch_size).view(-1))
                model.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                for p in model.parameters():
                    p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)


# Number of errors
def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        output_0 = model(input.narrow(1, 0, 1).narrow(0, b, mini_batch_size))
        output_1 = model(input.narrow(1, 1, 1).narrow(0, b, mini_batch_size))
        _, predicted_classes_0 = output_0.data.max(1)
        _, predicted_classes_1 = output_1.data.max(1)

        diffs = predicted_classes_1-predicted_classes_0
        diffs[diffs >= 0] = 1
        diffs[diffs < 0] = 0
        for k in range(mini_batch_size):
            if diffs[k] != target[b+k]:
                nb_errors = nb_errors + 1

    return nb_errors


# Code exectution

for mini_batch_size in [10, 20, 50, 100]:
    print('Mini batch size {:d}'.format(mini_batch_size))
    model = Classifier_Net()
    for k in range(1):
        train_model(model, train_input, train_classes, mini_batch_size)
        nb_classification_errors = compute_nb_errors(model, train_input,
                                                     train_target, mini_batch_size)
        print('Train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_classification_errors) / test_input.size(0),
                                                          nb_classification_errors, test_input.size(0)))
        nb_classification_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        print('Test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_classification_errors) / test_input.size(0),
                                                          nb_classification_errors, test_input.size(0)))

