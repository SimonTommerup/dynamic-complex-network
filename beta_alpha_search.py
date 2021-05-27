# %%
import nodespace
import nhpp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
np.seterr(all='raise')

ns = nodespace.NodeSpace()
# n_clusts = 3
# n_points = [7, 7, 7]
# centers = [[2,2], [-3,-3], [3,3]]
# radius = [2.1,2.5,1.9]
# v = [[-1,0], [1,0], [0,-1]]
# a =  [[-0.1,0], [0.1,0], [0.01,0]]

n_clusts = 2
n_points = [10,10]
centers = [[-3,-3], [3,3]]
radius = [1.0, 1.0]
v = [[5,0], [-5,0]]
a = [[-1, 0], [1, 0]]

z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
ns.init_conditions(z0, v0, a0)

ns.beta=1
ns.alpha=1e-5

t = np.linspace(0, 15)
rmat = nhpp.root_matrix(ns)
mmat = nhpp.monotonicity_mat(ns, rmat)
nhppmat = nhpp.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)


# %%

test_u = 2
test_v = 11

e = nhppmat[test_u,test_v]
plt.hist(e)
plt.show()

print("Root at:", rmat[test_u,test_v])

lambda_int_0_T = ns.lambda_int_rapprox(t, test_u, test_v)
print("Expected value (no. events):", lambda_int_0_T)
print("Actual value (no. events):", len(e))

#%%

print(rmat.shape)
# %%
lens = []
ind = np.triu_indices(rmat.shape[0], k=1)

for u, v in zip(ind[0],ind[1]):
    lens.append(len(rmat[u,v]))

plt.hist(lens)
plt.show()
# %%
print(rmat)
# %%
print(lens)
# %%
