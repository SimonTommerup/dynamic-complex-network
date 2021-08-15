import torch
import nodespace
import smallnet_sqdist as smallnet
import os
import nhpp_mod
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
gt_net = smallnet.SmallNet(4, 7.5)
gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_net.load_state_dict(gt_dict)
gt_net = gt_net.eval()

#print([p for p in gt_net.named_parameters()])

# data
data_set = np.load("4-point-data-set.npy", allow_pickle=True)

# verify time ordering
prev_t = 0.
time_col=2
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t


data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
print("n_train:", n_train)
print("n_test:", len(test_data))

training_batches = np.array_split(training_data, 450)
batch_size = len(training_batches[0])
print("Batch size:", batch_size)


t0 = torch.tensor([0.0])
#net(batch, t0=start_t, tn=batch[-1][2])
monte_carlo_int = []
exact_int = []
ind = torch.triu_indices(row=4, col=4, offset=1)
n_samples = 1
times = [t0]
for idx, batch in enumerate(training_batches):
    print(f"Batch {idx} of {len(training_batches)}")
    tn = batch[-1][2]

    times.append(tn)

    mc_res = 0.0
    exact_res = 0.0
    for i,j in zip(ind[0], ind[1]):
        exact_res += gt_net.evaluate_integral(i, j, t0, tn, gt_net.z0, gt_net.v0, gt_net.beta)
        mc_res += gt_net.riemann_sum(i, j, t0, tn, n_samples)

    exact_int.append(exact_res)
    monte_carlo_int.append(mc_res)

    t0 = tn # move time to last in current batch

plt.plot(np.arange(len(training_batches)), exact_int, color="green", label="exact")
plt.plot(np.arange(len(training_batches)), monte_carlo_int, color="red", label="mc")
plt.legend(loc="lower left")
plt.xlabel("batch no")
plt.ylabel("Non-event intensity")
#plt.ylim(1400,2000)
plt.show()


#%%

print("Mean time jumps in batch:", np.mean(np.diff(np.array(times))))

ratios=[]
for e, m in zip(exact_int, monte_carlo_int):
    ratios.append(e/m)
print("mean ratio", sum(ratios)/len(ratios))
# %%


# %%
def rmp(x, p):
    p = p/100
    return x/(1+p)

rmp(45400,12)
# %%

func = lambda x: ((x*(x-1))/2)/(x-1)
xvals = np.arange(2,50)
plt.plot(xvals, func(xvals))
# %%
func(xvals)
# %%
2