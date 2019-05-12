# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:48:41 2019

@author: Charles
"""


import dlc_practical_prologue as prologue

import torch
import network
from torch import nn

import matplotlib.pyplot as plt


# ------------------------ Number of errors -----------------------------------
def compute_nb_errors(model, input, target, mini_batch_size = 100):
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
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
nb_epochs = 10
iterations = 5
nb_hidden = 50


models = [network.Sharing, network.noSharing]  # List of networks to be used


loss_history = torch.zeros(len(models)*2, nb_epochs*iterations)
nb_errors = torch.zeros(len(models)*2, iterations)

i = 0

for use_auxLoss in [True, False]:
    for model in models:
        model = model(nb_hidden)  # Initialisation of the network
        for n in range(iterations):
            loss_history[i, n*nb_epochs:(n+1)*nb_epochs] = model.train(
                    train_input, train_target, train_classes,
                    mini_batch_size, nb_epochs, use_auxLoss = use_auxLoss)
            nb_errors[i, n] = compute_nb_errors(model, test_input, test_target,
                                                mini_batch_size)
        i += 1

torch.save(loss_history, 'loss_history.pt')
torch.save(nb_errors, 'nb_errors.pt')

