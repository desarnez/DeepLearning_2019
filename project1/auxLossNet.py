# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:48:41 2019

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
class Comparison_Net(nn.Module):
    def __init__(self):
        super(Comparison_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*3*3, 50)
        self.fc1prime = nn.Linear(50,10)
        self.fc2 = nn.Linear(2*50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward_conv(self,x):
        x = F.tanh(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.fc1(x.view(-1, 64*3*3)))
        return x

    def forward_linear(self, x1, x2):
        x = torch.cat((x1,x2),1)
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x1 = x.narrow(1,0,1)
        x2 = x.narrow(1,1,1)
        x1 = self.forward_conv(x1)
        x1prime = self.fc1prime(x1)
        x2 = self.forward_conv(x2)
        x2prime = self.fc1prime(x2)
        x = self.forward_linear(x1, x2)
        return x, x1prime, x2prime



# Training function
def train_model(model, train_input, train_target, mini_batch_size):
    criterion_target = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    train_target_tmp = torch.full((train_target.size(0), 2), 0)
    train_target_tmp[train_target == 1, 1] = 1
    train_target_tmp[train_target == 0, 0] = 1
    train_target = train_target_tmp

    eta = 1e-1
    for e in range(15):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output, x1, x2 = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion_target(output, train_target.narrow(0, b, mini_batch_size)) + criterion_class(x1, train_classes.narrow(1,0,1).narrow(0, b, mini_batch_size).view(-1)) + criterion_class(x2, train_classes.narrow(1,1,1).narrow(0, b, mini_batch_size).view(-1))
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
        output,_,_ = model(input.narrow(0, b, mini_batch_size))
        _, predicted_comparison = output.data.max(1)
        for k in range(mini_batch_size):
            if predicted_comparison[k] != target[b+k]:
                nb_errors = nb_errors + 1
#        print(nb_errors)
    return nb_errors


# Code exectution


mini_batch_size = 100
print('Mini batch size {:d}'.format(mini_batch_size))
model = Comparison_Net()
for k in range(10):
    train_model(model, train_input, train_target, mini_batch_size)
    nb_classification_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
    print('Train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_classification_errors) / test_input.size(0),
                                                      nb_classification_errors, test_input.size(0)))
    nb_classification_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('Test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_classification_errors) / test_input.size(0),
                                                      nb_classification_errors, test_input.size(0)))

