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

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

# Create dynamical system with constant velocity
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 9.5

z_gt = np.array([[-3, 1.], [-3., -1.], [3.,1.], [3.,-1.]])
v_gt = np.array([[2.0,-2.0], [2.0,2.0], [-2.0,-2.0], [-2.0, 2.0]])
a_gt = np.array([[-0.5,0.6], [-0.5,-0.5], [0.5,0.5], [0.5, -0.5]])
n_points=len(z_gt)
ns_gt.init_conditions(z_gt, v_gt, a_gt)

# Simulate event time data set for the two nodes
t = np.linspace(0, 15)
rmat = nhpp.root_matrix(ns_gt) 
mmat = nhpp.monotonicity_mat(ns_gt, rmat)
nhppmat = nhpp.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

# create data set and sort by time
ind = np.triu_indices(n_points, k=1)
data_set = []
for u,v in zip(ind[0], ind[1]):
    event_times = nhpp.get_entry(nhppmat, u=u, v=v)
    for e in event_times:
        data_set.append([u, v, e])

data_set = np.array(data_set)
time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]

plt.hist(data_set[:,2])
plt.show()
plt.close()

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t


#np.save("4-point-new-acceleration-data-set.npy", data_set)
verify_data = np.load("4-point-new-acceleration-data-set.npy", allow_pickle=True)

print("Simulated: ", data_set[0])
print("Saved: ", verify_data[0])

data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.9)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)


plot_data = data_set.numpy()
plt.hist(plot_data[:,2])
plt.title("Data: Event histogram")
plt.vlines(x=training_data[-1][2].item(), ymin=0, ymax=15000, color="r")
plt.ylabel("Frequency")
plt.grid()
plt.xlim(0,10)
plt.ylim((0,15000))
plt.xticks(ticks=(np.arange(0,10)))
plt.xlabel("Time")
plt.show()
