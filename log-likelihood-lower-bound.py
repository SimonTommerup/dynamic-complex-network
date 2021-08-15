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

np.seterr(all='raise')

ns= nodespace.NodeSpace()
ns.beta = 7.5

total_count=0
success_count=0
for seed in range(0, 100):
    total_count += 1
    ZERO_SEED = seed
    np.random.seed(ZERO_SEED)
    torch.manual_seed(ZERO_SEED)
    np.seterr(all='raise')

    z0 = np.array([[-0.5, -0.5], [.5, .5]])
    v0 = np.array([[0.25,0.0], [-0.25,0]])
    a0 = np.array([[0.,0.], [0.,0.]] )
    n_points=len(z0)
    ns.init_conditions(z0, v0, a0)

    # Simulate event time data set for the two nodes
    t = np.linspace(0, 5)
    rmat = nhpp_mod.root_matrix(ns) 
    mmat = nhpp_mod.monotonicity_mat(ns, rmat)
    nhppmat = nhpp_mod.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

    # create data set and sort by time
    ind = np.triu_indices(n_points, k=1)
    data_set = []
    for u,v in zip(ind[0], ind[1]):
        event_times = nhpp_mod.get_entry(nhppmat, u=u, v=v)
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


    net = smallnet.SmallNet(2, 7.5, non_intensity_weight=1)
    state_dict = net.state_dict()
    net_z = torch.from_numpy(z0)
    net_v = torch.from_numpy(v0)
    state_dict["z0"] = net_z
    state_dict["v0"] = net_v
    net.load_state_dict(state_dict)
    net.eval()

    data_set = torch.from_numpy(data_set)

    outp, _ = net(data_set, 0, data_set[-1][2])
    print("GT on data set:", outp)


    net_closer = smallnet.SmallNet(2,7.5, non_intensity_weight=1)
    z_2 = np.array([[-0.49, -0.49], [0.49, 0.49]])
    v_2 = np.array([[0.25,0.0], [-0.25,0]])
    close_dict = net_closer.state_dict()
    close_z = torch.from_numpy(z_2)
    close_v = torch.from_numpy(v_2)
    close_dict["z0"] = close_z
    close_dict["v0"] = close_v
    net_closer.load_state_dict(close_dict)
    net_closer.eval()
    outp2, _ = net_closer(data_set, 0, data_set[-1][2])

    if outp2>outp:
        success_count += 1

#%%
def init_net(w, z, v):
    n=smallnet.SmallNet(2, 7.5, non_intensity_weight=w)
    sd = n.state_dict()
    sd["z0"] = torch.from_numpy(z)
    sd["v0"] = torch.from_numpy(v)
    n.load_state_dict(sd)
    return n


#%%
ZERO_SEED=14
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
weights = np.linspace(1,5,num=10)
true_loglik = []
cur_loglik = []
for w in weights:
    true_net = init_net(w=w, z=z0, v=v0)
    true_net.eval()

    cur_net = init_net(w=w, z=z_2, v=v_2)
    cur_net.eval()

    true_out, _ = true_net(data_set, 0, data_set[-1][2])
    output, _ = cur_net(data_set, 0, data_set[-1][2])
    true_loglik.append(true_out)
    cur_loglik.append(output)

plt.plot(weights, true_loglik, label="GT", color="blue")
plt.plot(weights, cur_loglik, label="Closer than GT", color="green")
plt.xticks(ticks=weights)
plt.legend(loc="lower left")
plt.ylabel("Log-likelihood")
plt.xlabel("Non-intensity weight")
plt.show()




#%%

plot_data = data_set.numpy()
plt.hist(plot_data[:,2])
plt.title("Data: Event histogram")
plt.ylabel("Frequency")
plt.grid()
plt.xticks(ticks=(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
plt.xlabel("Time")
plt.show()

# %%


plt.scatter(z0[:,0], z0[:,1], color="blue", label="GT")
plt.scatter(z_2[:,0], z_2[:,1], color="red", label="Closer than GT")
plt.axhline(y=-0.49,color="red")
plt.axhline(y=-0.5,color="blue")
plt.axhline(y=0.49,color="red")
plt.axhline(y=0.50,color="blue")
plt.title("Latent space positions and trajectory")
plt.legend()
plt.show()
# %%
