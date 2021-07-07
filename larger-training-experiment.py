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
data_set_path = "50point-newclust-beta=5" + "_data_set" + ".npy"
numpydata = np.load(data_set_path)

split_ratio = 0.8

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
batch_size = 128
train_loader = data.DataLoader(training_data, batch_size=batch_size)

#init_beta = torch.tensor([2.8085])
fpath = r"state_dicts/training_experiment"

gt_net = smallnet.SmallNet(n_points=50, init_beta=5., mc_samples=5, non_event_weight=1)
gt_net_path = "50-points-newclust-gtnet.pth"

gt_path = os.path.join(fpath, gt_net_path)
gt_net.load_state_dict(torch.load(gt_path))

track_nodes=[5,27]
tn_train = training_data[-1][2].item() # last time point in training data
tn_test = test_data[-1][2].item() # last time point in test data

def getres(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_fun(ti, track_nodes[0], track_nodes[1]))
    return torch.tensor(res)


def plotres(num_epochs, y_train, y_test, title):
    x = np.arange(num_epochs)
    plt.plot(x, y_train, "g", label="Train")
    plt.plot(x, y_test, "r", label="Test")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()
    plt.close()

def plotgrad(num_epochs, bgrad, zgrad, vgrad):
    x = np.arange(num_epochs)
    plt.plot(x, bgrad, "g", label="beta")
    plt.plot(x, zgrad, "b", label="z")
    plt.plot(x, vgrad, "r", label="v")
    plt.legend(loc="upper right")
    plt.title("Parameters: Mean abs grad for epoch")
    plt.show()
    plt.close()

res_gt=[getres(0, tn_train, gt_net, track_nodes), getres(tn_train, tn_test, gt_net, track_nodes)]    


NUM_EPOCHS = 50
NUM_INITS = 1
plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    #seed = ZERO_SEED + initialization
    seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    fname = f"batch={batch_size}_epochs={NUM_EPOCHS}_50_nodes_trackloss_weight=1_{seed}" + ".pth"
    full_path = os.path.join(fpath, fname)
    
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, mc_samples=5, non_event_weight=1)
    
    # if load previous
    #net.load_state_dict(torch.load(full_path))

    #net, train_loss, test_loss = smallnet.single_batch_train(net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    #net, train_loss, test_loss = smallnet.batch_train(net, n_train, training_data, train_loader, test_data, NUM_EPOCHS)
    net, train_loss, test_loss, track_dict = smallnet.batch_train_track_mse(res_gt, track_nodes, net, n_train, training_data, train_loader, test_data, NUM_EPOCHS)

    x_vals = np.arange(NUM_EPOCHS)

    plt.plot(x_vals, train_loss, "g", label="Train")
    plt.plot(x_vals, test_loss, "r", label="Test")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Loss")
    plt.show()
    plt.close()

    plt.plot(x_vals, track_dict["zgrad"], color="b", label="z")
    plt.plot(x_vals, track_dict["vgrad"], color="r", label="v")
    plt.plot(x_vals, track_dict["agrad"], color="m", label="a")
    plt.plot(x_vals, track_dict["bgrad"], color="g", label="beta")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute value")
    plt.title("Parameter gradient tracking")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    
    torch.save(net.state_dict(), full_path)

    
plt.plot(x_vals, track_dict["mse_train_losses"], label="Train")
plt.plot(x_vals, track_dict["mse_test_losses"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("Intensity rate function MSE")
plt.legend()
plt.show()
plt.close()
