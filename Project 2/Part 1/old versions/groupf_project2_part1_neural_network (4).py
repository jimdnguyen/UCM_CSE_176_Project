# -*- coding: utf-8 -*-
"""GroupF_Project2_Part1_Neural_Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E6NJe6Uk5KclG8jj7gOonhcTGmckyo0o
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer

random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
np.random.seed(random_seed)
g = torch.Generator()
g.manual_seed(random_seed)
#https://github.com/pytorch/pytorch/issues/7068
#https://discuss.pytorch.org/t/random-seed-initialization/7854/28
#https://pytorch.org/docs/stable/notes/randomness.html

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, 500)  # 28*28 from image dimension 
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 5)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
if torch.cuda.is_available():
  net = net.cuda()

print(net)

def data_loader(batch_size, n_workers):
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True)
    #test_data_th = datasets.MNIST(root='./datasets', download=True, train=False)

    label = [1, 2, 3 ,4, 6]
    data_fea = np.array(train_data_th.data[:]).reshape([-1, 28 * 28]).astype(np.float32)
    data_fea = (data_fea / 255)
    data_gnd = np.array(train_data_th.targets)
    ctr1_idx = np.where(data_gnd[:] == label[0])
    ctr2_idx = np.where(data_gnd[:] == label[1])
    ctr3_idx = np.where(data_gnd[:] == label[2])
    ctr4_idx = np.where(data_gnd[:] == label[3])
    ctr6_idx = np.where(data_gnd[:] == label[4])
    ctr1_idx = np.array(ctr1_idx)
    ctr1_idx = ctr1_idx[0,0:1500]
    ctr2_idx = np.array(ctr2_idx)
    ctr2_idx = ctr2_idx[0,0:1500]
    ctr3_idx = np.array(ctr3_idx)
    ctr3_idx = ctr3_idx[0,0:1500]
    ctr4_idx = np.array(ctr4_idx)
    ctr4_idx = ctr4_idx[0,0:1500]
    ctr6_idx = np.array(ctr6_idx)
    ctr6_idx = ctr6_idx[0,0:1500]
    train_idx = np.concatenate((ctr1_idx[:500], ctr2_idx[:500], ctr3_idx[:500], ctr4_idx[:500], ctr6_idx[:500]),axis = None)
    validation_idx = np.concatenate((ctr1_idx[500:1000], ctr2_idx[500:1000], ctr3_idx[500:1000], ctr4_idx[500:1000], ctr6_idx[500:1000]),axis = None)
    test_idx = np.concatenate((ctr1_idx[1000:1500], ctr2_idx[1000:1500], ctr3_idx[1000:1500], ctr4_idx[1000:1500], ctr6_idx[1000:1500]),axis = None)

    data_train = data_fea[train_idx]
    target_train = data_gnd[train_idx]

    data_validation = data_fea[validation_idx]
    target_validation = data_gnd[validation_idx]

    data_test = data_fea[test_idx]
    target_test = data_gnd[test_idx]

    ##not sure what this is doing but it was here in the og
    dtrain_mean = data_train.mean(axis=0)
    data_train -= dtrain_mean
    data_validation -=dtrain_mean
    data_test -= dtrain_mean
    ##

    #######
    #https://discuss.pytorch.org/t/indexerror-target-2-is-out-of-bounds/69614/24 I AM SO HAPPY I FOUND THIS FORUM!
    tensor_target_train = torch.from_numpy(target_train)
    # print(tensor_target_train.size())
    # print(min(tensor_target_train))
    # print(max(tensor_target_train))
    unique_targets_train = torch.unique(tensor_target_train)
    # print('unique_targets_train: {}'.format(unique_targets_train))

    new_tensor_target_train = torch.empty_like(tensor_target_train)
    for idx, t in enumerate(unique_targets_train):
        # print('replacing {} with {}'.format(t, idx))
        new_tensor_target_train[tensor_target_train == t] = idx
    # print(new_tensor_target_train.size())
    # print(min(new_tensor_target_train))
    # print(max(new_tensor_target_train))

    tensor_target_validation = torch.from_numpy(target_validation)
    unique_targets_validation = torch.unique(tensor_target_validation)
    new_tensor_target_validation = torch.empty_like(tensor_target_validation)
    for idx, t in enumerate(unique_targets_validation):
      new_tensor_target_validation[tensor_target_validation == t] = idx

    tensor_target_test = torch.from_numpy(target_test)
    unique_targets_test = torch.unique(tensor_target_test)
    new_tensor_target_test = torch.empty_like(tensor_target_test)
    for idx, t in enumerate(unique_targets_test):
      new_tensor_target_test[tensor_target_test == t] = idx

    #new size of dataset is 2500
    train_data = TensorDataset(torch.from_numpy(data_train), new_tensor_target_train)
    validation_data = TensorDataset(torch.from_numpy(data_validation), new_tensor_target_validation)
    test_data = TensorDataset(torch.from_numpy(data_test), new_tensor_target_test)

    train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True,worker_init_fn=seed_worker,generator=g)
    validation_loader = DataLoader(validation_data, num_workers = n_workers, batch_size = batch_size, shuffle = True,worker_init_fn=seed_worker,generator=g)
    test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False,worker_init_fn=seed_worker,generator=g)

    return train_loader, validation_loader, test_loader

def calc_acc(loader, net):
    correct_cnt = 0
    total_cnt = 0
    #net.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            if torch.cuda.is_available():
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()
            out = net(batch_inputs)
            _, pred_labels = torch.max(out.data, 1)
            total_cnt += batch_labels.size(0)
            correct_cnt += (pred_labels == batch_labels).sum().item()

    return correct_cnt / total_cnt

plotval = []
plottrain = []
plotdiftrainval = []
plotepoch = []
plotloss = []
avg_loss = []
timetaken = []

# batch_size_init = 2048
# n_workers_init = 2
# train_loader, validation_loader, test_loader = data_loader(batch_size_init, n_workers_init)

# print("Before optimizing the model")
# print(f'train accurary: {100 * calc_acc(train_loader,net):.3f}%')
# print(f'validation accurary: {100 * calc_acc(validation_loader,net):.3f}%')
# print(f'test accurary: {100 *calc_acc(test_loader,net):.3f}%')

nnloss = torch.nn.CrossEntropyLoss()
from torch import nn, optim

#I optimized lr, momentum, weight_decay, batch, epoch, step_size, and gamma
#values are seen in the code blocks below

# def train_net(i):
#     net = Net()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     # Your task is to choose best parameters of optimization (momentum, batch size, learning rate, l2 regularization).
#     params = list(filter(lambda p: p.requires_grad, net.parameters()))
#     optimizer = optim.SGD(params, lr=i)  # lr = learning rate, momentum = momentum, weight delay = l2 regulaization
#     #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12,gamma=0.8)
#     #net.train()
#     batch_size_init = 50
#     n_workers_init = 2
#     train_loader, validation_loader, test_loader = data_loader(batch_size_init, n_workers_init)
#     epochs = 10
#     tmpepoch = 0
        
#     for epoch in range(epochs):
#         tmpepoch = epoch
#         print(f"Learning Rate is {i}")
#         print(f"Epoch is {epoch}")
#         avg_loss = []
#         clearList()
#         for x, target in train_loader:
#             if torch.cuda.is_available():
#                 x = x.cuda()[:]
#                 target = target.cuda().to(dtype=torch.long)
#             else:
#                 x = x[:]
#                 target = target.to(dtype=torch.long)
#             optimizer.zero_grad()
#             out = net(x)
#             loss = nnloss(out, target)
#             avg_loss.append(loss.item())
#             loss.backward()
#             optimizer.step()
#         #scheduler.step()

#         # print(f"\tepoch #{epoch} is finished.")
#         plotepoch.append(epoch)
#         # print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")
#         plotloss.append(np.mean(avg_loss))
#         train_acc = calc_acc(train_loader, net)
#         print(f'Train Accurary: {train_acc}')
#         plottrain.append(train_acc)
#         validation_acc = calc_acc(validation_loader, net)
#         print(f'Validation Accurary: {validation_acc}')
#         plotval.append(validation_acc)
#         dif = train_acc - validation_acc
#         plotdiftrainval.append(dif)
#         print(f"Difference between Train and Val {dif}")
#     fig = plt.figure(1)
#     plt.semilogx(plotval, color="blue", label="Val_Score", marker="o")
#     plt.semilogx(plottrain, color="red", label="Train_Score", marker="o")
#     plt.title(f'LearningRate = {i} epoch = {tmpepoch} ')
#     plt.show()
#     fig = plt.figure(2)
#     plt.cla()
#     plt.semilogx(plotepoch,plotloss,color = "green", label = "loss vs epoch", marker = "o")
#     plt.show()
#     findAcc(train_loader, validation_loader, test_loader, net)
#     findIdx()
#     clearList()

"""The block of code down below is my attempts of trying to running the best learning rate. Currently, I am running into the issue of not understanding how to reset the parameters or even if i do need to reset the parameters
look at this [website](https://discuss.pytorch.org/t/reset-the-parameters-of-a-model/29839). they explain some things how to reset the parameters



"""

def clearList():
    plotval.clear()
    plottrain.clear()
    plotdiftrainval.clear()
    plotepoch.clear()
    plotloss.clear()
    avg_loss.clear()

def findIdx():
    #https://www.geeksforgeeks.org/python-remove-negative-elements-in-list/
    tmpplot = [ele for ele in plotdiftrainval if ele > 0]
    if len(tmpplot) != 0:
        minDif = min(tmpplot)
        mindDifidx = plotdiftrainval.index(minDif)
        print(f'Minimum Difference between Train Acc and Validation Acc: {minDif}')
        print(f'Index where Minimum Difference between Train Acc and Validation Acc: {mindDifidx}')
    
    maxVal = max(plotval)
    maxValidx = plotval.index(maxVal)
    print(f'Max value of validation accurary: {maxVal}')
    print(f'Index where Max value of validation accurary: {maxValidx}')

def findAcc(train_loader, validation_loader, test_loader, net):
    print("After optimizing the model")
    print(f'Train Accurary: {100 * calc_acc(train_loader,net):.3f}%')
    print(f'Validation Accurary: {100 * calc_acc(validation_loader,net):.3f}%')
    #print(f'test accurary: {100 *calc_acc(test_loader,net):.3f}%')

#train_net()

# lrate = [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1]
# lrate2 = np.linspace(0,1,101)
# #lrate3 = np.linspace(0,0.3,31)
# for i in range(len(lrate)) :
#     train_net(lrate[i])

# momentumrate = np.linspace(0.1,1,10)
# for i in range(len(momentumrate)) :
#     net = Net()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     train_net(net,momentumrate[i])

#second parameter optimized, found highest val at 0.9

# weight_decayrate = np.linspace(0,1,11)
# for i in range(len(weight_decayrate)) :
#     net = Net()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     train_net(net,weight_decayrate[i])
#third parameter optimized, found highest val accurary at 0

# stepsizerate = np.linspace(0,100,101)
# for i in range(len(stepsizerate)) :
#     net = Net()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     train_net(net,stepsizerate[i])
#sixth parameter optimized, found highest val accurary at 28

# gammarate = np.linspace(0,1,11)
# for i in range(len(gammarate)) :
#     net = Net()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     train_net(net,gammarate[i])
#seventh parameter optimized, found highest val accurary at 0

#train_net(net)

# this was the code used to calc batch and epoch 
def batch_epoch_train_net():
    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()
    # Your task is to choose best parameters of optimization (momentum, batch size, learning rate, l2 regularization).
    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.SGD(params, lr=0.1)  # lr = learning rate, momentum = momentum, weight delay = l2 regulaization
    batchnum = [2500,1250,625,250,125,100,50,25,20,10,4,2,1]
    epochrate = [1,2,4,10,20,25,50,100,125,250,625,1250,2500]
    #net.train()

    for i in batchnum:
        start = timer()
        batch_size_init = i
        n_workers_init = 2
        train_loader, validation_loader, test_loader = data_loader(batch_size_init, n_workers_init)
        tmpepoch = 0
        for epoch in epochrate:
            #print(f"\tBatchSize is {i}")
            #print(f"\tEpoch is {epoch}")
            for x, target in train_loader:
                if torch.cuda.is_available():
                    x = x.cuda()[:]
                    target = target.cuda().to(dtype=torch.long)
                else:
                    x = x[:]
                    target = target.to(dtype=torch.long)
                optimizer.zero_grad()
                out = net(x)
                loss = nnloss(out, target)
                avg_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print(f"\tepoch {epoch} is finished.")
            plotepoch.append(epoch)
            print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")
            plotloss.append(np.mean(avg_loss))
            train_acc = calc_acc(train_loader, net)
            #print(f'Train Accurary: {train_acc}')
            plottrain.append(train_acc)
            validation_acc = calc_acc(validation_loader, net)
            #print(f'Validation Accurary: {validation_acc}')
            plotval.append(validation_acc)
            dif = train_acc - validation_acc
            plotdiftrainval.append(dif)
            #print(f"Difference between Train and Val {dif}")    
        fig = plt.figure(1)
        plt.plot(plotval, color="blue", label="Val_Score", marker="o")
        plt.plot(plottrain, color="red", label="Train_Score", marker="o")
        plt.title(f'BatchSize = {i}')
        plt.legend(loc="lower right")
        plt.xlabel("Iterations over list of custom batch_size and epoch values")
        plt.ylabel("Accuracy Values")
        plt.show()
        fig2 = plt.figure(2)
        plt.plot(plotloss, color = "green", label = "epoch_vs_loss", marker = "o")
        plt.title(f'Epoch vs Loss')
        plt.legend(loc = "upper right")
        plt.xlabel("Iterations over list of custom epoch values")
        plt.ylabel("Avg Train Loss")
        plt.show()
        findAcc(train_loader, validation_loader, test_loader, net)
        findIdx()
        clearList()
        end = timer()
        taken=end-start
        print(f"It took us {taken} seconds to run this loop")
        timetaken.append(taken)

batch_epoch_train_net()