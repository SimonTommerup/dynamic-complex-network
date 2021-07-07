import torch
import nodespace
import smallnet_eucliddist as smallnet
import os
import nhpp
import compare_rates
import data_sets
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs

ZERO_SEED = 1
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

ns_gt = nodespace.NodeSpace()
ns_gt.beta = 5
n_points = 50

#%%

# z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
# v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])

#%%
centers = np.array([[-4.,0.0],[4.,0.],[4,-4.]])
z_gt, y = make_blobs(n_samples=n_points, centers=centers, center_box=(-10,10))

fig, ax = plt.subplots()
scatter = ax.scatter(z_gt[:,0], z_gt[:,1], c=y)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.title("Latent space")
plt.show()
plt.close()


#%%
#print(y)
# custom_init_dynamics(self, n_points, labels, vdir, adir):

vdir = np.array([[0.04, -0.04], [-0.02, -0.04], [-0.04, 0.02] ])
adir = np.array([[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001] ])

v_gt, a_gt = ns_gt.custom_init_dynamics(n_points, y, vdir, adir)
ns_gt.init_conditions(z_gt, v_gt, a_gt)

gt_net = smallnet.SmallNet(n_points=50, init_beta=5., mc_samples=5, non_event_weight=1)

gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_a = torch.from_numpy(a_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_dict["a0"] = gt_a

gt_net.load_state_dict(gt_dict)
dirpath = r"state_dicts/training_experiment"
gt_net_path = "50-points-newclust-gtnet.pth"
torch.save(gt_net.state_dict(), os.path.join(dirpath, gt_net_path))

#test_gt = smallnet.SmallNet(n_points=50, init_beta=5., mc_samples=5, non_event_weight=1)
#test_gt.load_state_dict(torch.load(os.path.join(dirpath, gt_net_path)))


rmat = nhpp.root_matrix(ns_gt) 
mmat = nhpp.monotonicity_mat(ns_gt, rmat)

t = np.linspace(0,15)

print("Simulating events...")
t0 = time.time()
nhppmat = nhpp.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
print("Elapsed simulation time (s): ", time.time() - t0)

#path = "50point-newclust-beta=5"
#npy_ext = ".npy"
# np.save(path + npy_ext, nhppmat)

#%%
test_u, test_v = 0, 49
events=nhpp.get_entry(nhppmat, u=test_u, v=test_v)
roots = nhpp.get_entry(rmat, u=test_u, v=test_v)
plt.hist(events)
plt.show()
plt.close()
print("Roots:", roots)


#%%
res=[]
for ti in t:
    res.append(ns_gt.lambda_fun(ti, test_u, test_v))

plt.plot(t, res)
plt.show()
plt.close()

#%%
gt_net.eval()
res=[]
for ti in t:
    res.append(gt_net.lambda_fun(ti, test_u, test_v))

plt.plot(t, res)
plt.show()
plt.close()


#%%
# create data set and save
#nhppmat = np.load(path + npy_ext, allow_pickle=True)

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

print(f"Data set len: {len(data_set)}")
# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t

data_set_path = path + "_data_set" + npy_ext
#data_set = np.save(data_set_path, data_set)
data_set = np.load(data_set_path, allow_pickle=True)
# #%%
# def get_specs(ns, u, v):
#     z0 = np.array([ns.z0[u,:], ns.z0[v,:]])
#     v0 = np.array([ns.v0[u,:], ns.v0[v,:]])
#     a0 = np.array([ns.a0[u,:], ns.a0[v,:]])
#     return z0, v0, a0

# tz, tv, ta = get_specs(ns_gt, 4, 106)
# # %%
# tz

# %%
plt.hist(data_set[:,2])
# %%

split_ratio = 0.8
n = len(data_set)
n_train = int(np.ceil(n*split_ratio))
n_test = n - n_train
print("n_train:", n_train)
print("n_test:", n_test)


tn_train = data_set[n_train][2]

train_time = np.linspace(0,tn_train)
