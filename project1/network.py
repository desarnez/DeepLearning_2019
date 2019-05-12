# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:42:19 2019

@author: Charles
"""
import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.fc1 = nn.Linear(24*3*3, 10)

    def forward(self, x):
        x = torch.tanh(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.fc1(x.view(-1, 24*3*3)))
        return x

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.fc1 = nn.Linear(24*3*3, 10)

    def forward(self, x):
        x = torch.tanh(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.fc1(x.view(-1, 24*3*3)))
        return x


class Comparator(nn.Module):
    def __init__(self, nb_hidden):
        super(Comparator, self).__init__()
        self.fc2 = nn.Linear(2*10, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)

    def forward(self, x1, x2):
        x = torch.tanh(self.fc2(torch.cat((x1, x2), 1)))
        x = self.fc3(x)
        return x


class noSharing(nn.Module):
    def __init__(self, nb_hidden):
        super(noSharing, self).__init__()
        self.classify1 = Classifier()
        self.classify2 = Classifier()
        self.compare = Comparator(nb_hidden)

    def forward(self, x):
        x1 = x.narrow(1, 0, 1)
        x2 = x.narrow(1, 1, 1)
        x1 = self.classify1(x1)
        x2 = self.classify2(x2)
        x = self.compare(x1,x2)
        return x, x1, x2

    def train(self, train_input, train_target, train_classes,
              mini_batch_size =100, nb_epochs = 20, lr = 1e-1,
              use_auxLoss = True):

        criterion = nn.CrossEntropyLoss()
        loss_history = torch.zeros(nb_epochs)

        for e in range(nb_epochs):
            sum_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                output, x1, x2 = self(train_input.narrow(0, b, mini_batch_size))
                if use_auxLoss is True:
                    loss = criterion(output, train_target.narrow(0, b, mini_batch_size))\
                    + criterion(x1, train_classes.narrow(1, 0, 1).narrow(0, b, mini_batch_size).view(-1))\
                    + criterion(x2, train_classes.narrow(1, 1, 1).narrow(0, b, mini_batch_size).view(-1))
                else:
                    loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

                self.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                for p in self.parameters():
                    p.data.sub_(lr * p.grad.data)
            loss_history[e] = sum_loss
        return loss_history

class Sharing(nn.Module):
    def __init__(self, nb_hidden):
        super(Sharing, self).__init__()
        self.classify = Classifier()
        self.compare = Comparator(nb_hidden)

    def forward(self, x):
        x1 = x.narrow(1, 0, 1)
        x2 = x.narrow(1, 1, 1)
        x1 = self.classify(x1)
        x2 = self.classify(x2)
        x= self.compare(x1,x2)
        return x, x1, x2


    def train(self, train_input, train_target, train_classes,
              mini_batch_size =100, nb_epochs = 20, lr = 1e-2,
              use_auxLoss = True):

        criterion = nn.CrossEntropyLoss()
        loss_history = torch.zeros(nb_epochs)

        for e in range(nb_epochs):
            sum_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                output, x1, x2 = self(train_input.narrow(0, b, mini_batch_size))
                if use_auxLoss is True:
                    loss = criterion(output, train_target.narrow(0, b, mini_batch_size))\
                    + criterion(x1, train_classes.narrow(1, 0, 1).narrow(0, b, mini_batch_size).view(-1))\
                    + criterion(x2, train_classes.narrow(1, 1, 1).narrow(0, b, mini_batch_size).view(-1))
                else:
                    loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

                self.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                for p in self.parameters():
                    p.data.sub_(lr * p.grad.data)
            loss_history[e] = sum_loss
        return loss_history



class dumb(nn.Module):
    def __init__(self, nb_hidden):
        super(dumb, self).__init__()
        self.conv1 = nn.Conv2d(2, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.fc1 = nn.Linear(24*3*3, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = torch.tanh(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.fc1(x.view(-1, 24*3*3)))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x