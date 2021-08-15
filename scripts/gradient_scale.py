import torch
import nodespace
import smallnet_eucliddist as smallnet
import os
import nhpp
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

ZERO_SEED = 7
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')


data_set = np.load("4-point-acceleration-data-set.npy", allow_pickle=True)
data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]

def map_to_range(arr,i):
    vmaxd = arr[-1]*i
    vmind = arr[0]
    vmax = arr[-1]
    vmin = vmind

    for idx, val in enumerate(arr):
        new_val = (vmaxd-vmind)/(vmax-vmin)*(val-vmin)+vmind
        arr[idx] = new_val
    return arr


n_points = 4
init_beta = 7.195
lr = 0.001

bgrad_size = []
zgrad_size = []
vgrad_size = []
batch_sizes= np.arange(0,60000,10000)
batch_sizes[0] = batch_sizes[0]+1
#batch_sizes=[120, 130, 140, 150, 160]
number_of_points = [4,16,32,64]
scale_time_interval = [1,2,4,6,8]

#%%

for i in number_of_points:
    print(f"experiment batch_size = {i}")
    batch=training_data[:128]

    #batch[:,2] = map_to_range(batch[:,2], i)

    tn_train = batch[-1][2]

    untrained_model = smallnet.SmallNet(3, 4, 3.0, 5, 4)

    untrained_model_dict = untrained_model.state_dict()
    untrained_model_dict["z0"] = torch.zeros(size=(i,2))
    untrained_model.load_state_dict(untrained_model_dict)

    _, track_dict = smallnet.single_batch_train_track_mse(untrained_model, lr, batch)

    bgrad_size.append(track_dict["bgrad"])
    zgrad_size.append(track_dict["zgrad"])
    vgrad_size.append(track_dict["vgrad"])

batch_sizes=number_of_points
plt.plot(batch_sizes, bgrad_size, color="red", label="beta")
plt.plot(batch_sizes, zgrad_size, color="blue", label="z")
plt.plot(batch_sizes, vgrad_size, color="green", label="v")
plt.ylabel("Gradient absolute value")
plt.xlabel("Number of points")
plt.legend(loc="upper left")
plt.show()






