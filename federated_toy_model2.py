"""
implementation of the Federated Average Algorithm
"""
# imports
import torch as th
import syft as sy
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch import nn, optim
from torch.functional import F


# transform the data set for normilization
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# download the data set
trainset = datasets.MNIST(
    "data_set/mnist_fed", download=True, train=True, transform=transform
)
valset = datasets.MNIST(
    "data_set/mnist_fed", download=True, train=False, transform=transform
)
# trainloader = th.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = th.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# split the dataset into 4 splits for 6 workers
(
    train_set_1,
    train_set_2,
    train_set_3,
    train_set_4,
    train_set_5,
    train_set_6,
) = th.utils.data.random_split(
    trainset, [10000, 10000, 10000, 10000, 10000, 10000]
)  # giviging 10000 data set to each of the worker
trainloader_1 = th.utils.data.DataLoader(
    train_set_1, batch_size=64, shuffle=True
)  # first of the train set split
trainloader_2 = th.utils.data.DataLoader(
    train_set_2, batch_size=64, shuffle=True
)  # second of the train set split
trainloader_3 = th.utils.data.DataLoader(
    train_set_3, batch_size=64, shuffle=True
)  # third of the train set split
trainloader_4 = th.utils.data.DataLoader(
    train_set_4, batch_size=64, shuffle=True
)  # fourth of the train set split
trainloader_5 = th.utils.data.DataLoader(
    train_set_5, batch_size=64, shuffle=True
)  # fourth of the train set split
trainloader_6 = th.utils.data.DataLoader(
    train_set_6, batch_size=64, shuffle=True
)  # fourth of the train set split

#data loader one
dataiter_1 = iter(trainloader_1)
images_1, labels_1 = dataiter_1.next()

print(images_1.shape)
print(labels_1.shape)

#data loader two
dataiter_2 = iter(trainloader_2)
images_2, labels_2 = dataiter_2.next()

print(images_2.shape)
print(labels_2.shape)

#data loader three
dataiter_3 = iter(trainloader_3)
images_3, labels_3 = dataiter_3.next()

print(images_3.shape)
print(labels_3.shape)

#data loader four
dataiter_4 = iter(trainloader_4)
images_4, labels_4 = dataiter_4.next()

print(images_4.shape)
print(labels_4.shape)

#data loader five
dataiter_5 = iter(trainloader_5)
images_5, labels_5 = dataiter_5.next()

print(images_5.shape)
print(labels_5.shape)

#data loader six
dataiter_6 = iter(trainloader_6)
images_6, labels_6 = dataiter_6.next()

print(images_6.shape)
print(labels_6.shape)


class Worker():
    '''
    class worker represents a virtual device,
    .....
    Attributes:
    .............
    trainloader :
        the data loading paramater
    parameter :
        the weights and the bias of the model

    Methods:
    ............
    train()
    '''
    def __init__(self,train_loader, model, loss_lst):
        self.train_loader = train_loader
        self.model = model
        self.loss_lst = loss_lst
    def train_function(self):
        '''
        for training of the model
        train : no arguments

        returns updated paramaters
        '''
        criterion = nn.NLLLoss()
        images, labels = next(iter(self.train_loader))
        images = images.view(images.shape[0], -1)
        logps = self.model(images) #log probabilities
        loss = criterion(logps, labels) #calculate the NLL loss
        optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        running_loss = 0
        for images, labels in self.train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()
                
            output = self.model(images)
            loss = criterion(output, labels)
                
            #This is where the model learns by backpropagating
            loss.backward()     
                
            #And optimizes its weights here
            optimizer.step()
                
        running_loss += loss.item()
        
        print("Epoch - Training loss: {}".format(running_loss/len(self.train_loader)))
        #print("\nTraining Time (in minutes) =",(time()-time0)/60)
        self.loss_lst.append(running_loss/len(self.train_loader))



#making a model
#making the global network for the trainig 
input_size = 784
hidden_sizes = [128, 64]
output_size = 10


model_0 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),  #model 0 is the global model
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
model_1 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
model_2 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

model_3 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

model_4 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

model_5 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

model_6 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

#training for a test 

loss_lst_1 = []
loss_lst_2 = []
loss_lst_3 = []
loss_lst_4 = []
loss_lst_5 = []
loss_lst_6 = []
#making the 6 workers
w_1 = Worker(trainloader_1, model_1, loss_lst_1)
w_2 = Worker(trainloader_2, model_2, loss_lst_2)
w_3 = Worker(trainloader_3, model_3, loss_lst_3)
w_4 = Worker(trainloader_4, model_4, loss_lst_4)
w_5 = Worker(trainloader_5, model_5, loss_lst_5)
w_6 = Worker(trainloader_5, model_6, loss_lst_6)

#training the model
epoch_lst = []
for i in range(40):
    print("epoch", i)
    epoch_lst.append(i)
    print(w_1.train_function())
    print(w_2.train_function())
    print(w_3.train_function())
    print(w_4.train_function())
    print(w_5.train_function())
    print(w_6.train_function())

print(epoch_lst)

def aggreagator():
    #initialize the model with first models
    pass


#testing if model is traing or not 

#checking the accuracy of a trained model 



def accuracy_check(model,valloader):
    correct_count, all_count = 0, 0
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with th.no_grad():
                logps = model(img)

            ps = th.exp(logps)

            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

print(accuracy_check(model_0, valloader))
print(accuracy_check(w_1.model, valloader))
print(accuracy_check(w_2.model, valloader))
print(accuracy_check(w_3.model, valloader))
print(accuracy_check(w_4.model, valloader))
print(accuracy_check(w_5.model, valloader))
print(accuracy_check(w_6.model, valloader))

#plotting the losss graph 

plt.plot( epoch_lst,w_1.loss_lst , label = "loss_1")


#plotting the graph 
plt.plot( epoch_lst,w_2.loss_lst, label = "loss_2")


#plotting the graph 
plt.plot( epoch_lst,w_3.loss_lst , label = "loss_3")


#plotting the graph 
plt.plot( epoch_lst,w_4.loss_lst , label = "loss_4")


#plotting the graph 
plt.plot( epoch_lst,w_5.loss_lst , label = "loss_5")


#plotting the graph 
plt.plot( epoch_lst,w_6.loss_lst , label = "loss_6")
plt.legend()
plt.show()

