import torch
import nodespace
import smallnet
import os
import nhpp_mod
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Set seeds
# Train 10 models with minibatch size = n_train
# Train 10 models with minibatch size = 70
# Save the models, save the losses

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

# Create dynamical system with constant velocity
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 7.5

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
n_points=len(z_gt)
ns_gt.init_conditions(z_gt, v_gt, a_gt)


# Simulate event time data set for the two nodes
t = np.linspace(0, 1)
rmat = nhpp_mod.root_matrix(ns_gt) 
mmat = nhpp_mod.monotonicity_mat(ns_gt, rmat)
nhppmat = nhpp_mod.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)


# create data set and sort by time
ind = np.triu_indices(n_points, k=1)
data_set = []
for u,v in zip(ind[0], ind[1]):
    event_times = nhpp_mod.get_entry(nhppmat, u=u, v=v)
    print("Len event times:", len(event_times))
    for e in event_times:
        data_set.append([u, v, e])

data_set = np.array(data_set)
time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t


data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

print("n_train:", len(training_data))
print("n_test:", len(test_data))

init_beta = smallnet.infer_beta(n_points, training_data)

NUM_EPOCHS = 1
NUM_INITS = 10

plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    seed = ZERO_SEED + initialization 
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta)

    net, train_loss, test_loss = smallnet.single_batch_train(net, num_train_samples, training_data, test_data, NUM_EPOCHS)

    plt.plot(np.arange(NUM_EPOCHS), train_loss, "g")
    plt.plot(np.arange(NUM_EPOCHS), test_loss, "r")
    plt.show()
    plt.close()


    fpath = r"state_dicts/training_experiment"
    fname = f"batch=n_train_model_init_{seed}" + ".pth"

    torch.save(net.state_dict(), os.path.join(fpath, fname))
