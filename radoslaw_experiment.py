import torch
import nodespace
import smallnet_node_specific_beta as smallnet # CHECK MODEL
import os
import nhpp
import data_sets
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import sklearn.model_selection


DATA_PATH = "radoslaw-email.npy"
data_set = np.load(DATA_PATH, allow_pickle=True)

# verify time ordering
prev_t = 0.
time_col=2
for row in data_set:
    cur_t = row[time_col]
    assert cur_t >= prev_t
    prev_t = cur_t

n_points = 167

data_set = torch.from_numpy(data_set)
n_train = int(len(data_set)*0.8)
train_data = data_set[0:n_train]
test_data = data_set[n_train:]

n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)


#init_beta = torch.nn.init.uniform_(torch.zeros(size=(1,1)), 0.0, 0.025)

training_batches = np.array_split(train_data, 450)
#training_batches = np.array_split(train_data, 2)

batch_size = len(training_batches[0])
print("Batch size:", batch_size)

node_pair_samples=750

init_beta = 4.86
model = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)

#lfpath = r"state_dicts/training_experiment"
#lfname = f"radoslaw-bs148-nps=750-gc30-smpl=1-node-specific-beta" + ".pth"
#lfull_path = os.path.join(lfpath, lfname)
#model.load_state_dict(torch.load(lfull_path))

model.beta.requires_grad = True
model.z0.requires_grad = True
model.v0.requires_grad = True
model.a0.requires_grad = True

num_epochs=50

model, train_loss, test_loss = smallnet.batch_train_scheduled(model, n_train, train_data, training_batches, test_data, num_epochs)


fpath = r"state_dicts/training_experiment"
fname = f"radoslaw-bs148-nps=750-gc30-smpl=1-node-specific-beta" + ".pth"
full_path = os.path.join(fpath, fname)
torch.save(model.state_dict(), full_path)

x_epochs = np.arange(1,num_epochs+1)
plt.plot(x_epochs, train_loss, label="Train", color="red")
plt.plot(x_epochs, test_loss, label="Test", color="green")
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.legend()
plt.show()
plt.close()

#%%

n_points=167
init_beta=4.86
lfpath = r"state_dicts/training_experiment"
lfname = f"radoslaw-bs148-nps=750-gc30-smpl=1-node-specific-beta" + ".pth"
full_path = os.path.join(lfpath, lfname)
trained_model = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)
trained_model.load_state_dict(torch.load(full_path))

#%%

est_z0 = trained_model.z0.detach()
est_z0 = est_z0.numpy()


plt.scatter(est_z0[:,0], est_z0[:,1], color="red", label="est")
plt.legend()
plt.title("Z latent space")
plt.show()

# %%
