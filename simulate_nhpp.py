#%%
from nodespace import NodeSpace
import numpy as np
import torch
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


ns = NodeSpace()
n_clusts = 3
n_points = [1, 1, 1]
centers = [[2,-1], [-3,0], [3,3]]
radius = [0,0,0]
v = [[0,1], [1,0], [0,-1]]
a =  [[0,0], [0,0], [0,0]]

z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
ns.init_conditions(z0, v0, a0)

print(ns.z0)

# %%
# Objective: Simulate n(n-1)/2 nhpp at a time

# Find roots
def root_matrix(ns):
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    rmat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        z_ux, z_uy = ns.z0[u, 0], ns.z0[u, 1]
        z_vx, z_vy = ns.z0[v, 0], ns.z0[v, 1]

        # phi-substitutions
        p1 = ns.v0[u,0] - ns.v0[v, 0] 
        p2 = ns.a0[u,0] - ns.a0[v, 0]
        p3 = ns.v0[u,1] - ns.v0[v, 1]
        p4 = ns.a0[u,1] - ns.a0[v, 1]

        # coefficients
        a = (p4**2 + p2**2)/2
        b = (3*p1*p2 + 3*p3*p4)/2
        c = p2*(z_ux - z_vx) + p4*(z_uy -z_vy) + p1**2 +p3**2
        d = p1*(z_ux - z_vx) + p3*(z_uy - z_vy)

        r = np.roots([a,b,c,d])
        r = r[np.isreal(r)]
        r = np.sort(r)
        
        rmat[u, v] = r
    return rmat

rmat = root_matrix(ns)
print(rmat)
# %%
## Monotonicity
# 
def monotonicity_mat(ns, root_matrix):
    beta = ns.beta
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    mmat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        i = 0
        roots = root_matrix[u,v]
        time_points = [roots[0] - 0.5] # t to get sign of dLambda(t)/dt before first root
        while i < len(roots)-1:
            t = roots[i] + (roots[i+1] - roots[i]) / 2 # t's to get sign of dLambda(t)/dt between roots
            time_points.append(t)
            i += 1 
        time_points.append(roots[-1] + 0.5) # t to get sign of dLambda(t)/dt after last root

        monotonicity = []
        for t in time_points:
            val = ns.lambda_ddt(t, u, v) # value of derivative of lambda(t,u,v) at t

            if val < 0:
                monotonicity.append("dec")
            elif val > 0:
                monotonicity.append("inc")

        mmat[u,v] = np.array(monotonicity)

    return mmat

mmat = monotonicity_mat(ns, rmat)


# %%
print(mmat)
# %%
