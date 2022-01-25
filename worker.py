import torch as th
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

class Worker():
    """
    class worker represents a virtual device,
    .....
    Attributes:
    .............
    trainloader :
        the data loading paramater
    model :
        the DNN model
    loss_lst:
        a list to store the model losses
    Methods:
    ............
    train()

    arguents : model

    return : trained model with updated paramater

    """
    #input_size = 784
    #hidden_sizes = [128, 64]
    #output_size = 10

    #def __init__(self, train_loader, loss_lst,model = model_0):
    def __init__(self, train_loader, loss_lst,model):
        #print("Initilaizer in class ",model.output.bias)
        self.train_loader = train_loader
        self.loss_lst = loss_lst
        self.model = model

    #def train_function(self,model = model_0):
    def train_function(self):
        #self.model = model
        print("training ----------------------- starts ----------------------- ")
        print("before training model paramater", self.model.output.bias )
        """
        for training of the model
        train : no arguments

        returns updated paramaters
        """
        criterion = nn.NLLLoss()
        images, labels = next(iter(self.train_loader))
        images = images.view(images.shape[0], -1)
        logps = self.model(images)  # log probabilities
        loss = criterion(logps, labels)  # calculate the NLL loss
        optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        running_loss = 0
        for images, labels in self.train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()

            output = self.model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

        running_loss += loss.item()

        print("Epoch - Training loss: {}".format(running_loss / len(self.train_loader)))
        # print("\nTraining Time (in minutes) =",(time()-time0)/60)
        self.loss_lst.append(running_loss / len(self.train_loader))
        print("model bias after training",self.model.output.bias)  #check for model bias

        return (self.model.input.weight , self.model.hidden.weight , self.model.output.weight , 
        self.model.input.bias , self.model.hidden.bias , self.model.output.bias)  

class Worker_2():
    """
    class worker represents a virtual device,
    .....
    Attributes:
    .............
    trainloader :
        the data loading paramater
    model_2 :
        the DNN model_2
    loss_lst_2:
        a list to store the model_2 losses
    Methods:
    ............
    train()

    arguents : model_2

    return : trained model_2 with updated paramater

    """
    #input_size = 784
    #hidden_sizes = [128, 64]
    #output_size = 10

    #def __init__(self, train_loader_2, loss_lst_2,model_2 = model_0):
    def __init__(self, train_loader_2, loss_lst_2,model_2):
        #print("Initilaizer in class ",model_2.output.bias)
        self.train_loader_2 = train_loader_2
        self.loss_lst_2 = loss_lst_2
        self.model_2 = model_2

    #def train_function_2(self,model_2 = model_0):
    def train_function_2(self):
        #self.model_2 = model_2
        print("training ----------------------- starts ----------------------- ")
        print("before training model_2 paramater", self.model_2.output.bias )
        """
        for training of the model_2
        train : no arguments

        returns updated paramaters
        """
        criterion = nn.NLLLoss()
        images_2, labels_2 = next(iter(self.train_loader_2))
        images_2 = images_2.view(images_2.shape[0], -1)
        logps = self.model_2(images_2)  # log probabilities
        loss = criterion(logps, labels_2)  # calculate the NLL loss
        optimizer = optim.SGD(self.model_2.parameters(), lr=0.003, momentum=0.9)
        running_loss = 0
        for images_2, labels_2 in self.train_loader_2:
            # Flatten MNIST images_2 into a 784 long vector
            images_2 = images_2.view(images_2.shape[0], -1)
            # Training pass
            optimizer.zero_grad()

            output = self.model_2(images_2)
            loss = criterion(output, labels_2)

            # This is where the model_2 learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

        running_loss += loss.item()

        print("Epoch - Training loss: {}".format(running_loss / len(self.train_loader_2)))
        # print("\nTraining Time (in minutes) =",(time()-time0)/60)
        self.loss_lst_2.append(running_loss / len(self.train_loader_2))
        print("model_2 bias after training",self.model_2.output.bias)  #check for model_2 bias

        return (self.model_2.input.weight , self.model_2.hidden.weight , self.model_2.output.weight , 
        self.model_2.input.bias , self.model_2.hidden.bias , self.model_2.output.bias)  