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
from sklearn.datasets import make_blobs

STATE_DICT_PATH = r"state_dicts/training_experiment"
ZERO_SEED = 1
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

# Create dynamical system 
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 3.55
n_points=150
centers = np.array([[-5.,-3],[4.,4.],[4,-3.], [-5,5]])
z_gt, y = make_blobs(n_samples=n_points, centers=centers, center_box=(-10,10))

fig, ax = plt.subplots()
scatter = ax.scatter(z_gt[:,0], z_gt[:,1], c=y)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Cluster")
ax.add_artist(legend1)
plt.xlim(-12, 8)
plt.title("Z Latent space")
plt.show()
plt.close()
 
vdir = np.array([[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0],  [1.0, -1.0]])*2
adir = np.array([[-0.1, -0.1], [0.1, 0.1], [0.1, -0.1], [-0.1, 0.1] ])*5
v_gt, a_gt = ns_gt.custom_init_dynamics(n_points, y, vdir, adir)
ns_gt.init_conditions(z_gt, v_gt, a_gt)

# Simulate event time data set for sampled nodes
# n_node_pairs = 150*149//2
# node_pair_samples  = n_node_pairs // 5
# ind = np.triu_indices(n_points, k=1)
# triu_samples = torch.randperm(n_node_pairs)[:node_pair_samples]
# triu_samples = triu_samples.numpy()

# t = np.linspace(0, 10)
# rmat = nhpp.root_matrix(ns_gt)
# nhppmat = nhpp.nhpp_mat_sampled(ns=ns_gt, samples=triu_samples, time=t, root_matrix=rmat)

# create data set and sort by time
# data_set = []
# for sample_idx in triu_samples:
#     u, v = ind[0][sample_idx], ind[1][sample_idx] 
#     event_times = nhpp.get_entry(nhppmat, u=u, v=v)
#     for e in event_times:
#         data_set.append([u, v, e])

# data_set = np.array(data_set)
# time_col = 2
# data_set = data_set[data_set[:,time_col].argsort()]
#np.save("150-point-data-set-b=3.55-t=10.npy", data_set)
data_set = np.load("150-point-data-set-b=3.55-t=10.npy", allow_pickle=True)
print("n_events:", len(data_set))

# verify time ordering
prev_t = 0.
time_col=2
for row in data_set:
    cur_t = row[time_col]
    assert cur_t >= prev_t
    prev_t = cur_t

# plt.hist(data_set[:,2])
# plt.show()
# plt.close()

data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.85)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)

# plot_data = data_set.numpy()
# plt.hist(plot_data[:,2])
# plt.title("Data: Event histogram")
# plt.vlines(x=training_data[-1][2].item(), ymin=0, ymax=15000, color="r")
# plt.ylabel("Frequency")
# plt.grid()
# plt.xlim(0,10)
# plt.ylim((0,18000))
# plt.xticks(ticks=(np.arange(0,11)))
# plt.xlabel("Time")
# plt.show()


training_batches = np.array_split(training_data, 575) 
# split=575 => bs=148

batch_size = len(training_batches[0])
print("Batch size:", batch_size)

node_pair_samples=740
init_beta = torch.nn.init.uniform_(torch.zeros(size=(1,1)), a=0, b=1)

model = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)

# load previous iteration
load_fname = f"synthetic-150-bs=148-gc=30-pretrain-beta-zv" + ".pth"
load_fpath = os.path.join(STATE_DICT_PATH, load_fname)
#model.load_state_dict(torch.load(load_fpath))

model.beta.requires_grad = True
model.z0.requires_grad = True
model.v0.requires_grad = True
model.a0.requires_grad = True

num_epochs=50
#model, train_loss, test_loss = smallnet.batch_train(model, n_train, training_data, training_batches, test_data, num_epochs)
model, train_loss, test_loss, best_state = smallnet.batch_train_scheduled(model, n_train, training_data, training_batches, test_data, num_epochs)


save_fname = f"synthetic-150-bs=148-gc=30-pretrain-beta-zva" + ".pth"
save_fpath = os.path.join(STATE_DICT_PATH, save_fname)
#torch.save(model.state_dict(), save_fpath)
torch.save(best_state, save_fpath)

#####
# PRETRAINING: 
# No annealing.
# Train Beta where Z=0, V=0, A=0
# Train Beta, Z where V=0, A=0
# Train Beta, Z, V where A=0

#####
# TRAINING:
# Train all pars with annealing.
# Keep best: I.e. lowest point before overfitting commences.


x_epochs = np.arange(1,num_epochs+1)
plt.plot(x_epochs, train_loss, label="Train", color="red")
plt.plot(x_epochs, test_loss, label="Test", color="green")
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.legend()
plt.show()
plt.close()