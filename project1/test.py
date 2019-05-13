# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:48:41 2019

@author: Charles
"""


import dlc_practical_prologue as prologue

import torch
import network
from torch import nn
import time
import matplotlib.pyplot as plt


# ------------------------ Number of errors -----------------------------------
def compute_nb_errors(model, input, target, mini_batch_size = 100):
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        if model.__str__() is 'Dumb':
            output = model(input.narrow(0, b, mini_batch_size))
        else:
            output, _, _ = model(input.narrow(0, b, mini_batch_size))

        _, predicted_comparison = output.data.max(1)

        for k in range(mini_batch_size):
            if predicted_comparison[k] != target[b+k]:
                nb_errors = nb_errors + 1
    return nb_errors


# Initialisation
train_input, train_target, train_classes, \
    test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

# Definition of execution parameters
mini_batch_size = 100
nb_epochs = 20
iterations = 10
nb_hidden = 50

# List of networks to be used
models = [network.noSharing, network.Sharing, network.Dumb]

# Initialisation of tensors to extract the performance data from the learning
loss_history = torch.zeros(len(models)*2-1, nb_epochs*iterations)
nb_errors = torch.zeros(len(models)*2-1, iterations)

i = 0
for p in models:
    for use_auxLoss in [True, False]:
        t1 = time.time()
        model = p(nb_hidden)  # Initialisation of the network
        print('\nNetwork: {:s}'.format(str(model)))
        print('Auxiliary Losses: {:8s}'.format(str(use_auxLoss)))
        for n in range(iterations):
            loss_history[i, n*nb_epochs:(n+1)*nb_epochs] = model.train(
                    train_input, train_target, train_classes,
                    mini_batch_size, nb_epochs, use_auxLoss = use_auxLoss)
            nb_errors[i, n] = compute_nb_errors(model, test_input, test_target,
                                                mini_batch_size)
            print('Number of epochs : {:d}   Test error rate : {:.2f}%'.format((n+1)*nb_epochs, 100*nb_errors[i, n].item()/test_input.size(0)))
        if p is network.Dumb:
            break
        t2 = time.time()-t1
        print('execution time : {:.2f}'.format(t2))
        i += 1



