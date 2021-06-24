import torch
import nodespace
import smallnet
import os
import nhpp
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

ns_gt = nodespace.NodeSpace()
ns_gt.beta = 5
n_points = 200

#%%

#%%

z_gt, y = make_blobs(n_samples=n_points, centers=3, center_box=(-5,5))
plt.scatter(z_gt[:,0], z_gt[:,1], c=y)
plt.title("Latent space")
plt.show()
plt.close()
#%%

v_gt, a_gt = ns_gt.rand_init_dynamics(n_points)
ns_gt.init_conditions(z_gt, v_gt, a_gt)

rmat = nhpp.root_matrix(ns_gt) 
mmat = nhpp.monotonicity_mat(ns_gt, rmat)

t = np.linspace(0,15)

print("Simulating events...")
t0 = time.time()
nhppmat = nhpp.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
print("Elapsed simulation time (s): ", time.time() - t0)

#np.save("200point-event-beta=5.npy", nhppmat)

#%%
# create data set and save
nhppmat = np.load("200point-event-matrix-beta=5.npy", allow_pickle=True)

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

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t

data_set = np.save("200point-data_set.npy", data_set)

#%%
def get_specs(ns, u, v):
    z0 = np.array([ns.z0[u,:], ns.z0[v,:]])
    v0 = np.array([ns.v0[u,:], ns.v0[v,:]])
    a0 = np.array([ns.a0[u,:], ns.a0[v,:]])
    return z0, v0, a0

tz, tv, ta = get_specs(ns_gt, 4, 106)
# %%
tz

# %%
