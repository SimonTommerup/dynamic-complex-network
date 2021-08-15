import torch
import nodespace
import smallnet_eucliddist as smallnet
#import smallnet_node_specific_beta as smallnet # CHECK MODEL
import os
import nhpp
import data_sets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

FPATH = r"state_dicts/training_experiment"
EXT = ".pth"

# import models
norm = "batch=141-epoch=30-gradclip=30-riemann=5-std=0.025-init=2" + EXT
unif = "batch=141-epoch=30-gradclip=30-riemann=5-unif=0.025-init=2" + EXT
normpath = os.path.join(FPATH, norm)
unifpath = os.path.join(FPATH, unif)

# args
pm = 3
n_points = 4
init_beta = 0.0
rs = 5
nps = 4*3/2
args = [pm, n_points, init_beta, rs, nps]

normmodel = smallnet.SmallNet(*args)
unifmodel = smallnet.SmallNet(*args)

normmodel.load_state_dict(torch.load(normpath))
unifmodel.load_state_dict(torch.load(unifpath))

# est_z0 = model.z0.detach().numpy()
# est_v0 = model.v0.detach().numpy()

# plt.scatter(est_z0[:,0], est_z0[:,1], color="black", label="All")
# plt.scatter(est_z0[top_nodes,0], est_z0[top_nodes,1], color="blue", label="Most connected")
# plt.scatter(est_z0[low_nodes,0], est_z0[low_nodes,1], color="red", label="Least connected")
# plt.legend(loc="lower center")
# plt.title("Z latent space with velocity")
# for node in low_nodes:
#     plt.annotate(str(node), (est_z0[node,0], est_z0[node,1]))
#     #plt.quiver(est_z0[node,0], est_z0[node,1], est_v0[node,0], est_v0[node,1])

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.001,0.002], [-0.002,-0.001], [0.002,-0.002], [-0.002, 0.001]])

def get_est(model):
    z0 = model.z0.detach().numpy()
    v0 = model.v0.detach().numpy()
    a0 = model.a0.detach().numpy()
    return z0, v0, a0

z0_unif, v0_unif, a0_unif = get_est(unifmodel)
z0_norm, v0_norm, a0_norm = get_est(normmodel)

plt.scatter(z_gt[:,0], z_gt[:,1], color="green", label="GT")
plt.scatter(z0_unif[:,0], z0_unif[:,1], color="red", label="U(-0.025, 0.025)")
plt.scatter(z0_norm[:,0], z0_norm[:,1], color="blue", label="N(0, 0.025)")

for node in range(n_points):
    plt.quiver(z_gt[node, 0], z_gt[node,1], a_gt[node,0], a_gt[node,1])
    plt.quiver(z0_unif[node, 0], z0_unif[node,1], a0_unif[node,0], a0_unif[node,1])
    plt.quiver(z0_norm[node, 0], z0_norm[node,1], a0_norm[node,0], a0_norm[node,1])

plt.axis("equal")
plt.title("Z latent space with accelerations to scale")
plt.legend()
plt.xlim((-0.75, 0.75))
plt.grid()
plt.show()



