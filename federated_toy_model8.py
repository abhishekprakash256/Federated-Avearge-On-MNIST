import torch as th
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

from models import NeuralNetwork
from worker import Worker, Worker_2

#load the data



# transform the data set for normilization
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
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

# split the dataset into 6 splits for 6 workers
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
trainloader_0 = th.utils.data.DataLoader(
    train_set_1, batch_size=64, shuffle=True
)  # first of the train set split
trainloader_1 = th.utils.data.DataLoader(
    train_set_2, batch_size=64, shuffle=True
)  # second of the train set split
trainloader_2 = th.utils.data.DataLoader(
    train_set_3, batch_size=64, shuffle=True
)  # third of the train set split
trainloader_3 = th.utils.data.DataLoader(
    train_set_4, batch_size=64, shuffle=True
)  # fourth of the train set split
trainloader_4 = th.utils.data.DataLoader(
    train_set_5, batch_size=64, shuffle=True
)  # fourth of the train set split
trainloader_5 = th.utils.data.DataLoader(
    train_set_6, batch_size=64, shuffle=True
)  # fourth of the train set split




loss_lst_0 = []
model_0 = NeuralNetwork()

model_1 = NeuralNetwork()

w_1 = Worker(trainloader_1,loss_lst_0, model_0)

loss_lst_1 = []
w_2 = Worker_2(trainloader_2,loss_lst_1, model_1)



#print(w_1.model.output.bias)

epoch_lst = [] #lst to store the epochs

def aggreagator():
    '''
    function aggregator :

    Arguments :
    0
    -----------------
    returns :
    -------------------
    the aggregated model weight for each of the worker

    '''

    # the worker will have the initial model but different data

    for i in range(5):
        print("epoch number", i)
        print("start for w_1")
        new_weights_bias_1 = w_1.train_function()
        print("end")
        print("start for w_2")
        new_weights_bias_2 = w_2.train_function_2()
        print("end")
        # aggregating the weights of the model

        # for checking purpose of the updates

        #First layer update
        weight_1 = nn.Parameter(
            (
                new_weights_bias_1[0]
                + 
                new_weights_bias_2[0]

            )
            / 2
        )

        #end
        
        #sencond layer update
        weight_2 = nn.Parameter(
            (
                new_weights_bias_1[1]
                + new_weights_bias_2[1]

            )
            / 2
        )

        #end
        #third layer update
        weight_3 = nn.Parameter(
            (
                new_weights_bias_1[2]
                +  new_weights_bias_2[2]

            )
            / 2
        )

        #end

        # bias aggregation for the model
                #First layer update
        bias_1 = nn.Parameter(
            (
                new_weights_bias_1[3]
                + new_weights_bias_2[3] 

            )
            / 2
        )


        #end
        
        #sencond layer update
        bias_2 = nn.Parameter(
            (
                new_weights_bias_1[4]
                + new_weights_bias_2[4]

            )
            / 2
        ) 



        #end
        #third layer update
        bias_3= nn.Parameter(
            (
                new_weights_bias_1[5]
                + new_weights_bias_2[5]

            )
            / 2
        )

        w_1.model.hidden.weight = weight_1
        w_2.model_2.hidden.weight = weight_1
        
        w_1.model.hidden.weight = weight_2
        w_2.model_2.hidden.weight = weight_2
        
        w_1.model.output.weight = weight_3
        w_2.model_2.output.weight = weight_3
        
        w_1.model.input.bias = bias_1
        w_2.model_2.input.bias = bias_1
        
        w_1.model.hidden.bias = bias_2
        w_2.model_2.hidden.bias = bias_2
        
        w_1.model.output.bias = bias_3
        w_2.model_2.output.bias = bias_3



        #end

        print("model after aggregation  for w_1", w_1.model.output.bias)
        print("model after aggregation for w_2", w_2.model_2.output.bias)

        
        epoch_lst.append(i)


aggreagator()



def accuracy_check(model_, valloader_):
    '''
    accuracy_check function take two arguments:
    model , valloader
    returns : the accuracy of the model on the test data set'''
    
    correct_count, all_count = 0, 0
    for images, labels in valloader_:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with th.no_grad():
                logps = model_(img)

            ps = th.exp(logps)

            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


model = NeuralNetwork()
print(model.output.bias)


print(accuracy_check(w_1.model, valloader))

print(accuracy_check(w_2.model_2, valloader))


print(accuracy_check(model, valloader))


