# %%
from simulation import NodeSpace
import numpy as np
import torch
from scipy.spatial.distance import pdist

# Objective:
# 1. Simulate non-homogeneous Poisson process (NHPP) for one node pair.

# 1.1 Initial conditions for two seperate nodes
ns = NodeSpace()
rad = 0

# 1.1.1 Positions
z0_u = ns.init_points(1, [1,1], rad)
z0_v = ns.init_points(1, [12,1], rad)
z0 = np.concatenate((z0_u, z0_v))

# 1.1.2 Velocities
v0_u = ns.init_velocity(1, 3, 0)
v0_v = ns.init_velocity(1, -3, 0)
v0 = np.concatenate((v0_u, v0_v))

# 1.1.3 Accelerations
a0_u = ns.init_acceleration(1, -0.5, 0)
a0_v = ns.init_acceleration(1, 0.5, 0)
a0 = np.concatenate((a0_u,a0_v))

# 1.1.4 Init
ns.init_cond(z0, v0, a0)


# %%
# 1.2 Preparation 
# 1.2.1 Time interval T=20, i = 0,..,9
t = np.linspace(0,20,10)

# 1.2.2 Find upper bound of intensities and intensity phase (decrease/increase) 
def ub_int(nodespace, t, beta=1.5e1):
    ls = []
    for ti in t:
        z = ns.step(ti)
        cur_dist = pdist(z, metric="euclidean")
        max_l = np.max(np.exp(beta - cur_dist))
        ls.append(max_l)
    return ls

def r_int(nodespace):
    zx = nodespace.z0[:,0]
    zy = nodespace.z0[:,1]
    vx = nodespace.v0[:,0]
    vy = nodespace.v0[:,1]
    ax = nodespace.a0[:,0]
    ay = nodespace.a0[:,1]

    p1 = vx[0]-vx[1]
    p2 = ax[0]-ax[1]
    p3 = vy[0]-vy[1]
    p4 = ay[0]-ay[1]
    
    a = (p4**2 + p2**2)/2
    b = (3*p1*p2 + 3*p3*p4) / 2
    c = p2*(zx[0]-zx[1]) + p4*(zy[0]-zy[1]) + p1**2 +p3**2
    d = p1*(zx[0]-zx[1]) + p3*(zy[0]-zy[1])

    r = np.roots([a,b,c,d])
    r = r[np.isreal(r)]
    r = np.sort(r)
    return r

def mon(nodespace, r, beta=1.5e1):
    i = 0
    ts = [r[0] - 1]
    while i < len(r)-1:
        t = r[i] + (r[i+1] - r[i]) / 2
        ts.append(t)
        i += 1
    ts.append(r[-1] + 1)

    def ddt(nodespace, t, beta):
        z = nodespace.step(t)
        z_uv = z[0,:] - z[1,:]
        dz_uv = (ns.v0[0,:]-ns.v0[1,:]) + (ns.a0[0,:]-ns.a0[1,:])*t
        dist = np.sqrt(z_uv[0]**2 + z_uv[1]**2)
        return -np.exp(beta-dist)*np.dot(z_uv, dz_uv) / dist

    mon = []
    for t in ts:
        evalf = ddt(nodespace, t, beta=2)
        if evalf > 0:
            mon.append("inc")
        elif evalf < 0:
            mon.append("dec")
    return mon

def rmon(nodespace):
    r = r_int(nodespace)
    m = mon(nodespace, r)
    return r, m

# %%
# Test case with simple initial conditions:
# a) The two nodes approach
# b) The two nodes pass eachother and continues
# c) The two nodes slows down to zero, then approaches eachother
# d) The two nodes passes eachother and separates indefinitely.
# This means it is expected that monotonicity outputs:

# "inc", "dec", "inc", "dec"
r, m = rmon(ns)
print("Roots:", r)
print("Monotonicity:", m)


## %%
# 1.3 NHPP piecewise thinning algorithm

# %%
