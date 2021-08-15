import torch
import nodespace
#import smallnet_eucliddist as smallnet
import smallnet_node_specific_beta as smallnet # CHECK MODEL
import os
import nhpp
import data_sets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


DATA_PATH = "radoslaw-email.npy"
data_set = np.load(DATA_PATH, allow_pickle=True)

n_points=167
node_pair_samples=695
init_beta = 4.86
n_train = int(len(data_set)*0.8)


train_data = data_set[0:n_train]
test_data = data_set[n_train:]

n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)


def floatcount(indices, data_set):
    counts = []
    for idx in indices:
        bin_data = data_set[np.logical_or(data_set[:,0]==idx, data_set[:,1]==idx)]
        count = [idx, len(bin_data)]
        counts.append(count)
    return np.array(counts)

def descending_degree(edge_counts, k):
    return np.argsort(edge_counts[:,1])[::-1][:k]

def ascending_degree(edge_counts,k):
    return np.argsort(edge_counts[:,1])[:k]

def paircount(u, v, data_set):
    uv_cond = np.logical_or(data_set[:,0]==u, data_set[:,1]==v)
    vu_cond = np.logical_or(data_set[:,0]==v, data_set[:,1]==u)
    bin_data = data_set[np.logical_or(uv_cond, vu_cond)]
    count = len(bin_data)
    return count



indices = np.arange(n_points)
edge_counts = floatcount(indices, train_data)
top_nodes = descending_degree(edge_counts, 10)
low_nodes = ascending_degree(edge_counts,20)
# %%


fpath = r"state_dicts/training_experiment"
#fname = f"radoslaw-bs148-gc30-smpl=1-train-all-lrschedul" + ".pth"
fname = f"radoslaw-bs148-nps=750-gc30-smpl=1-node-specific-beta" + ".pth"
full_path = os.path.join(fpath, fname)

model = smallnet.SmallNet(3, n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)
model.load_state_dict(torch.load(full_path))

model.eval()

t0 = torch.zeros(size=(1,1))
tn = torch.ones(size=(1,1))
a = model.riemann_sum(0,1,t0,tn,1)
b = model.riemann_sum(1,0,t0,tn,1)
print(a)
print(b)

#%%
est_z0 = model.z0.detach().numpy()
est_v0 = model.v0.detach().numpy()

plt.scatter(est_z0[:,0], est_z0[:,1], color="black", label="All")
plt.scatter(est_z0[top_nodes,0], est_z0[top_nodes,1], color="blue", label="Most connected")
plt.scatter(est_z0[low_nodes,0], est_z0[low_nodes,1], color="red", label="Least connected")
plt.legend(loc="lower center")
plt.title("Z latent space with velocity")
for node in low_nodes:
    plt.annotate(str(node), (est_z0[node,0], est_z0[node,1]))
    #plt.quiver(est_z0[node,0], est_z0[node,1], est_v0[node,0], est_v0[node,1])

plt.show()


# V = np.array([[1,1], [-2,2], [4,-7]])
# origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

# plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
# plt.show()
# %%

plt.quiver(est_z0[node,0], est_z0[node,1], est_v0[node,0], est_v0[node,1])
plt.axis('equal')
#plt.xticks([i for i in range(-15,15)])
#plt.yticks([i for i in range(-15,15)])
#plt.grid()
plt.show()
# %%
u_p = 159
v_p = 166
u_n = 163
v_n = 156

comb = [u_p, v_p, u_n, v_n]

for u in comb:
    for v in comb:
        print(f"u={u}, v={v}")
        print(f"Count: {paircount(u,v, train_data)}")

# %%
