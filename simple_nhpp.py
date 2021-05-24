# %%
from simulation import NodeSpace
import numpy as np
import torch
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

BETA_VALUE = 5

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
t = np.linspace(0,15,1000)

# 1.2.2 Find upper bound of intensities and intensity phase (decrease/increase) 
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

def mon(nodespace, r, beta=BETA_VALUE):
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
        evalf = ddt(nodespace, t, beta=beta)
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
# %%
def upperbounds(nodespace, time, roots, monotonicity, beta=BETA_VALUE):
    lambda_i = []
    mon = monotonicity
    rcount=0
    idx = 0
    while idx < len(time)-1:
        if mon[idx] != mon[idx + 1]:
            ztr = ns.step(roots[rcount])
            dist = pdist(ztr, metric="euclidean")
            lambda_tr = np.exp(beta - dist)
            if mon[idx] == "dec":
                zt = ns.step(time[idx+1])
                dist = pdist(zt, metric="euclidean")
                lambda_tnext = np.exp(beta - dist)
            
            max_lambda = np.maximum(lambda_tr, lambda_tnext)
            lambda_i.append(max_lambda)
            rcount += 1
        
        elif mon[idx] == "inc":
            zt = ns.step(time[idx+1])
            dist = pdist(zt, metric="euclidean")
            lambda_tnext = np.exp(beta - dist)
            lambda_i.append(lambda_tnext)
        elif mon[idx] == "dec":
            zt = ns.step(time[idx])
            dist = pdist(zt, metric="euclidean")
            lambda_tcur = np.exp(beta - dist)
            lambda_i.append(lambda_tcur)

        idx += 1
    return np.array(lambda_i)

# map time points to monotonicity
def ttomon(time, root, mon):
    rprev = -np.inf
    cur_r, cur_m, last_m = 0, 0, False
    phases = []
    for t in time:
        if (rprev < t) & (t < r[cur_r]):
            phases.append(mon[cur_m])
        elif last_m:
            phases.append(mon[cur_m+1])
        elif (r[cur_r] < t) & (t < r[cur_r+1]):
            rprev = r[cur_r]
            cur_r += 1
            cur_m += 1
            if cur_r == len(r) - 1:
                last_m = True
            phases.append(mon[cur_m])
    return phases

phases= ttomon(t, r, m)


# %%
# 1.3 NHPP piecewise thinning algorithm
def nhpp(nodespace, time, upperbounds, beta=BETA_VALUE):
    interval = 0
    t = 0
    stop = False
    event_times = []
    exceed_interval = False
    exceed_count = 0
    not_exceed_count = 0
    iteration = 0

    # run until stop
    while not stop:
        iteration += 1
        
        # generate exponential rv
        if not exceed_interval:
            #print("here")
            u1 = np.random.uniform(0,1)
            x = -1 / upperbounds[interval] * np.log(u1)

        if (t + x) < time[interval + 1]:
            not_exceed_count += 1
            exceed_interval = False

            t += x
            u2 = np.random.uniform(0,1)

            # accept t if u2 < lambda(t) / lambda upperbound
            # compute lambda(t)
            zt = nodespace.step(t)
            d = pdist(zt, metric="euclidean")
            lambda_t = np.exp(beta - d)
            assert lambda_t < upperbounds[interval], "Lambda value exceeds upperbound"

            prob = lambda_t / upperbounds[interval]
            assert prob <= 1, "Probability out of range >1"
            assert prob >= 0, "Probability out of range <0"

            if u2 < prob:
                event_times.append(t)

        elif (t + x) >=  time[interval + 1]:
            exceed_count += 1
            exceed_interval = True
            if interval == len(upperbounds) - 1:
                stop = True
            else:
                x = (x - time[interval + 1] + t) * upperbounds[interval] / upperbounds[interval+1]
                t = time[interval + 1]
                interval += 1

    return np.array(event_times), exceed_count, not_exceed_count, iteration

def nhpp_global_bound(ns, time, beta=BETA_VALUE):
    t, i = 0, 0
    event_times = []
    iteration = 0
    ub_l = np.exp(beta)
    while t < time[-1]:
        iteration += 1
        u1 = np.random.uniform(0,1)
        t = t - (1/ub_l)*np.log(u1)
        zt = ns.step(t)
        dist = pdist(zt, metric="euclidean")
        t_l = np.exp(beta - dist)
        u2 = np.random.uniform(0,1)
        if u2 <= (t_l / ub_l):
            i += 1
            event_times.append(t)
    return event_times, i, iteration



# %%

r, m = rmon(ns)
time2monoticity = ttomon(t, r, m)
lambda_upper = upperbounds(ns, t, r, time2monoticity)
#e = nhpp_global(ns, t, lambda_upper, beta=5)


# %%
events, i, ite = nhpp_global_bound(ns, t, beta=BETA_VALUE)
print("i", i)
print("ite", ite)

print("Accept per iteration %", (i / ite) * 100 )

#%%

event_times, ec, nec, ite = nhpp(ns, t, lambda_upper, beta=BETA_VALUE)

print("ec", ec)
print("nec", nec)
print("ite", ite)
print("ec+nec", ec+nec)
print("No events:", len(event_times))
print("Accept per iteration %", (len(event_times) / ite) * 100 )

# %%
plt.hist(event_times)
plt.show()

# %%
print(len(event_times))
# %%
