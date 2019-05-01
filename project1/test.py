# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:36:10 2019

@author: Charles
"""

import dlc_practical_prologue as prologue

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(256, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        print(x.size())
        x = F.tanh(F.max_pool2d(self.conv2(x), kernel_size=2))
        print(x.size())
        x = F.tanh(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


###############################################################################
        
def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.NLLLoss() 
    eta = 1e-1 

    for e in range(20):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).long())
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

###############################################################################
        
def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

###############################################################################
def process_classes(classes):
    out_classes = torch.full((classes.size(0), 10), 0)
    for n in range(classes.size(0)):
        out_classes[n, classes[n]] = 1
    return out_classes

mini_batch_size = 100
 
train_input, train_target , train_classes, \
test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

train_input = train_input.narrow(1,0,1)
test_input = test_input.narrow(1,1,1)

#train_target = process_classes(train_classes.narrow(1,0,1))
#test_target = process_classes(test_classes.narrow(1,0,1))

train_target = train_classes.narrow(1,0,1)
test_target = test_classes.narrow(1,0,1)

model = Net()
for k in range(10):
    train_model(model,train_input, train_target[:,0], mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))

just making useless changes
to check thing

