from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Process, Array, Manager
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Accumulate loss
        running_loss += loss.item()
        
    '''Fill your code'''
    training_acc = 100. * correct / total
    training_loss = running_loss / len(train_loader) # Here, training_loss is loss per item. If you want loss per batch, then you can directly return running_loss
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    testing_acc, testing_loss = 100. * correct / len(test_loader.dataset), test_loss / len(test_loader.dataset)
    return testing_acc, testing_loss


def plot(epoches, performance, s):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    image_name = os.path.join(r'./figs/', s +'.png')
    plt.plot(epoches, performance, label = s)
    plt.xlabel('Epoch')
    plt.ylabel(s)
    plt.legend()
    plt.title(s)
    plt.savefig(image_name)
    #plt.show()
    #plt.close()
    plt.clf()


def run(config, seed):
    train_history = []
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    # i did this in another way, there's no need to modify this function.
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        epoches.append(epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        
        test_acc, test_loss = test(model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        #Print the step
        print("seed: {}, epoch: {}, train_acc: {}, train_loss: {}, test_acc: {}, test_loss: {}"
            .format(seed, epoch, train_acc, train_loss, test_acc, test_loss)
        )
        
        scheduler.step()
        """update the records, Fill your code"""
        # I updated the record outside of the for loop.
        """end filling"""
    train_history.append(epoches)
    train_history.append(training_accuracies)
    train_history.append(training_loss)
    train_history.append(testing_accuracies)
    train_history.append(testing_loss)
    
    """plotting training performance with the records"""
    plot(epoches, training_accuracies, "Training Accuracies With Seed" + str(seed))
    plot(epoches, training_loss, "Training Loss With Seed" + str(seed))

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "Testing Accuracies With Seed" + str(seed))
    plot(epoches, testing_loss, "Testing Loss With Seed" + str(seed))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
    return train_history


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""
    mean_training_accuracies = []
    mean_training_loss = []
    mean_testing_accuracies = []
    mean_testing_loss = []
    epoches = recorded[0][0]
    for i in range(0, recorded[0][0].__len__()):
        mean_training_accuracies.append((recorded[0][1][i]+recorded[1][1][i]+recorded[2][1][i]) / 3)
        mean_training_loss.append((recorded[0][2][i]+recorded[1][2][i]+recorded[2][2][i]) / 3)
        mean_testing_accuracies.append((recorded[0][3][i]+recorded[1][3][i]+recorded[2][3][i]) / 3)
        mean_testing_loss.append((recorded[0][4][i]+recorded[1][4][i]+recorded[2][4][i]) / 3)
        
    """plotting training performance with the records"""
    plot(epoches, mean_training_accuracies, "Mean Training Accuracies")
    plot(epoches, mean_training_loss, "Mean Training Loss")

    """plotting testing performance with the records"""
    plot(epoches, mean_testing_accuracies, "Mean Testing Accuracies")
    plot(epoches, mean_testing_loss, "Mean Testing Loss")
        
    
def run_all(config, seed, recorded, rank):
    res = run(config, seed)
    recorded.append(res)
    rank.append(seed)
    


if __name__ == '__main__':
    global recorded
    recorded = Manager().list()
    rank = Manager().list()
    arg = read_args()
    seeds = [123, 321, 666]
    processes = []
    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    
    print("Training starts.")
    time_start = time.time()
    
    for i in range(3):
        processes.append(
            Process(target = run_all, args=(config, seeds[i], recorded, rank))
        )
    
    for i in range(3):
        processes[i].start()
        
    for i in range(3):
        processes[i].join()
    
    time_end = time.time()
    print("Training ends. Time used: " + str(time_end - time_start))
    print("Finishing up...")
    
    records = list(recorded)
    """plot the mean results"""
    plot_mean()
    
    print(records)

    # Write records into a pandas dataframe than output
    columns = ['epochs', 'training accuracy', 'training loss', 'testing accuracy', 'testing loss']
    dataframes = [pd.DataFrame(list(map(list, zip(*record))), columns = columns) for record in records]
    
    writer = pd.ExcelWriter('data.xlsx')
    
    for i in range(3):
        dataframes[i].to_excel(writer, sheet_name = "seed{}".format(rank[i]))
        
    writer.save()
    

    # plot_mean()

    
