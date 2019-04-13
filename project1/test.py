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

class Net1(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(2, 5, kernel_size = (2,2) )
        self.conv2 = nn.Conv2d(5, 1, kernel_size = )



train_input, train_target , train_classes, \
test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

train_input.fill_(1, 1)


