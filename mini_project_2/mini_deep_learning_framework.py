#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:26:17 2019

@author: tristan.trebaol

"""

import torch
torch.set_grad_enabled(False)

class Module(object):
    
    def forward(self, *input): 
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    
    def param(self, *parameters): 
        
        return parameters
    
    # TODO : subclass for layer with weights
    def update_weigths(self, gradwrtoutput, in_features):
        
        return []


class Linear(Module):
    
    def _init_(self, in_neurons, out_neurons):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        # w(i,j) is the weight linking output i to input j
        self.weights = torch.emtpy(self.out_neurons, self.in_neurons)
        self.bias = self.out_neurons
        self.out_features = self.out_neurons
        
    def update_weigths(self, gradwrtoutput, in_features):
        # weigth update dl/dw
        inT = torch.transpose(in_features,0,1) #transpose of in_features vector
        eta = 0.01
        # gradwrtoutput is a column vector, inT is a line vector
        # weight(i,j) = weight(i,j) + eta * dl/d_layer_(i) * x_layer-1_(j)
        return self.weights + eta * torch.mv(gradwrtoutput, inT)
    
    def forward(self, in_features):
        # returning the output features
        self.out_features = torch.mv(self.weights, in_features) + self.bias
        return self.out_features
    
    def backward(self, gradwrtoutput, in_features):
        # need to 1. return the gradwrtinput, 2. update the inner weights,
        
        # 1. computing gradiant with respect to output dl/dx 
        # CONFIRM THAT WE TAKE THE WEIGHTS BEFORE UPDATE
        #  weight matrix is transposed
        wT = torch.transpose(self.weights,0,1)
        # gradwrtinput is the sum of dl/d_layer(i)*weights(i,j)
        gradwrtinput = torch.mv(wT, gradwrtoutput)
        
        # 2. updating the weights
        self.weights = self.update_weigths(gradwrtoutput, in_features)

        return gradwrtinput
    

    
class Tanh(Module):
        
    def forward(self, in_features):
        # this operation is applied element by element
        return torch.tanh(in_features)
    
    def backward(self, gradwrtoutput):
        return (1-torch.pow(torch.tanh(self.out_features), 2)) * gradwrtoutput
    
class ReLU(Module):
    
    def forward(self, in_features):
        # in_features should be a column vector
        # returns max(0,x)
        return max(torch.empty(in_features,1),in_features)
    
    def backward(self, gradwrtoutput):
        # x<0 => retunr 0
        # x>= 0 return 1
        return 
    