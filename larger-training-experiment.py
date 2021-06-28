import torch
import nodespace
import smallnet_eucliddist as smallnet
import os
import nhpp
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs
from data_sets import SyntheticData
from torch.utils import data

# Load data set
data_set_path = "50point-event-beta=5" + "_data_set" + ".npy"
numpydata = np.load(data_set_path)

split_ratio = 0.975

data_set = SyntheticData(numpydata=numpydata)

n = len(data_set)
n_train = int(np.ceil(n*split_ratio))
n_test = n - n_train
print("n_train:", n_train)
print("n_test:", n_test)

indices = [idx for idx in range(n)]
train_indices = indices[:n_train]
test_indices = indices[n_train:]

n_points = 50
init_beta = smallnet.infer_beta(n_points, data_set[:n_train])
print("init_beta:", init_beta) 

training_data = data.Subset(data_set, train_indices)
test_data = data.Subset(data_set, test_indices)
batch_size = 256
train_loader = data.DataLoader(training_data, batch_size=batch_size)




#init_beta = torch.tensor([2.8085])


NUM_EPOCHS = 20
NUM_INITS = 1
plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    #seed = ZERO_SEED + initialization
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)

    fpath = r"state_dicts/training_experiment"
    fname = f"batch={batch_size}_epochs={NUM_EPOCHS}_50_nodes_montecarlo_integral_init_{seed}" + ".pth"
    full_path = os.path.join(fpath, fname)
    
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, mc_samples=5)
    
    # if load previous
    #net.load_state_dict(torch.load(full_path))

    #net, train_loss, test_loss = smallnet.single_batch_train(net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    net, train_loss, test_loss = smallnet.batch_train(net, n_train, training_data, train_loader, test_data, NUM_EPOCHS)

    plt.plot(np.arange(NUM_EPOCHS), train_loss, "g", label="Train")
    plt.plot(np.arange(NUM_EPOCHS), test_loss, "r", label="Test")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    
    torch.save(net.state_dict(), full_path)

    
