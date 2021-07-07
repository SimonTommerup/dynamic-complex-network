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
n_points = 50

indices = [idx for idx in range(n)]
train_indices = indices[:n_train]
test_indices = indices[n_train:]


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

track_nodes=[0,5]
tn_train = training_data[-1][2].item() # last time point in training data
tn_test = test_data[-1][2].item() # last time point in test data

net_to_test = smallnet.SmallNet(n_points=50, init_beta=5., mc_samples=5, non_event_weight=1)
net_to_test_fname = f"batch=128_epochs=50_50_nodes_trackloss_weight=1_5" + ".pth"
net_to_test_path = os.path.join(fpath, net_to_test_fname)
net_to_test.load_state_dict(torch.load(net_to_test_path))


gt_net.eval()
net_to_test.eval()


def getres(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_fun(ti, track_nodes[0], track_nodes[1]))
    return torch.tensor(res)

def plotres(t0, tn, y_gt, y_est, title):
    x = np.linspace(t0, tn)
    plt.plot(x, y_gt, "g", label="gt")
    plt.plot(x, y_est, "r", label="est")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()
    plt.close()

def plotgt(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_fun(ti, track_nodes[0], track_nodes[1]))
    
    plt.plot(time, res, "g", label="gt")
    plt.title("just gt")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()

y_gt_train = getres(0, tn_train, gt_net, track_nodes)
y_est_train = getres(0, tn_train, net_to_test, track_nodes)
y_gt_test = getres(tn_train, tn_test, gt_net, track_nodes)
y_est_test = getres(tn_train, tn_test, net_to_test, track_nodes)

plotres(0, tn_train, y_gt_train, y_est_train, "Intensity: Training data")
plotres(tn_train, tn_test, y_gt_test, y_est_test, "Intensity: Test data")


plotgt(0, tn_test, gt_net, track_nodes)