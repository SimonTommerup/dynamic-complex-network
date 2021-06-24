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
numpydata = np.load("200point-data_set.npy")

split_ratio = 0.9

data_set = SyntheticData(numpydata=numpydata)

n = len(data_set)
n_train = int(np.ceil(n*split_ratio))
n_test = n - n_train
print("n_train:", n_train)
print("n_test:", n_test)

indices = [idx for idx in range(n)]
train_indices = indices[:n_train]
test_indices = indices[n_train:]

training_data = data.Subset(data_set, train_indices)
test_data = data.Subset(data_set, test_indices)

train_loader = data.DataLoader(training_data, batch_size=4096)

n_points = 200
# Method below yields init_beta = 2.8085 (gt=5), but may be too costly.
#init_beta = smallnet.infer_beta(n_points, training_data)
#print("init_beta:", init_beta) 
init_beta = torch.tensor([2.8085])


NUM_EPOCHS = 10
NUM_INITS = 1
plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    #seed = ZERO_SEED + initialization
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)



    fpath = r"state_dicts/training_experiment"
    fname = f"batch=141_LR=0.001_new_data_handle_montecarlo_integral_init_{seed}" + ".pth"
    full_path = os.path.join(fpath, fname)
    
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, mc_samples=1)
    
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



