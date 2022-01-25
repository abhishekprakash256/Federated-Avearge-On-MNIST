# imports
import torch as th
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to 1st hidden layer linear transformation 
        self.input = nn.Linear(784, 128)
        self.relu_0 = nn.ReLU()

        # hidden layer
        self.hidden = nn.Linear(128, 64)
        
        self.relu_1 = nn.ReLU()

        # Output layer 
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)

        
    def forward(self, x):
        x = self.input(x) 
        x = self.relu_0(x)
        x = self.hidden(x)
        x = self.relu_1(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x