#%%
from nodespace import NodeSpace
import numpy as np
import torch
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(1)

ns = NodeSpace()
n_clusts = 3
n_points = [7, 7, 7]
centers = [[2,-1], [-3,0], [3,3]]
radius = [2.1,2.5,1.9]
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
        r = r[np.isreal(r)] # real solutions
        r = r[r >= 0]       # positive time axis
        r = np.sort(r)
        
        rmat[u, v] = r
    return rmat

rmat = root_matrix(ns)

# %%

def monotonicity_mat(ns, root_matrix):
    beta = ns.beta
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    mmat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        i = 0
        roots = root_matrix[u,v]

        # If no real, non-negative roots, then the intensity rate is
        # either constant or monotonously increasing on positive time axis
        if len(roots) == 0:
            mmat[u,v] = np.array([])
            continue

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

# map time points to monotonicity
def time_to_monotonicity(time, roots, monotonicity):
    rprev = -np.inf
    cur_r, cur_m = 0, 0
    last_m = False if len(roots) > 1 else True
    ttm = []

    if len(monotonicity)==0:
        return ttm
        
    for t in time:
        if (rprev < t) & (t < roots[cur_r]):
            ttm.append(monotonicity[cur_m])
        elif last_m:
            ttm.append(monotonicity[cur_m+1])
        elif (roots[cur_r] < t) & (t < roots[cur_r+1]):
            rprev = roots[cur_r]
            cur_r += 1
            cur_m += 1
            if cur_r == len(roots) - 1:
                last_m = True
            ttm.append(monotonicity[cur_m])
    return ttm

# find intensity upperbounds for all time intervals
def upperbounds(ns, u, v, time, roots, time_to_monotonicity):
    beta = ns.beta
    lambda_arr = []
    mon = time_to_monotonicity
    rcount=0
    idx = 0

    if len(mon)==0:
        while idx < len(time)-1:
            cur_lambda = ns.lambda_fun(time[idx], u, v)
            next_lambda = ns.lambda_fun(time[idx+1], u, v)
            lambda_arr.append(np.maximum(cur_lambda, next_lambda))
            idx += 1
    else:
        while idx < len(time)-1:
            if mon[idx] != mon[idx + 1]:
                lambda_root = ns.lambda_fun(roots[rcount], u, v)

                if mon[idx] == "dec":
                    lambda_next_t = ns.lambda_fun(time[idx+1], u, v)
                
                max_lambda = np.maximum(lambda_root, lambda_next_t)
                lambda_arr.append(max_lambda)
                rcount += 1
            
            elif mon[idx] == "inc":
                lambda_next_t = ns.lambda_fun(time[idx+1], u, v)
                lambda_arr.append(lambda_next_t)

            elif mon[idx] == "dec":
                lambda_cur_t = ns.lambda_fun(time[idx], u, v)
                lambda_arr.append(lambda_cur_t)

            idx += 1

    lambda_arr = np.array(lambda_arr)
    return lambda_arr

def nhpp(ns, u, v, time, upperbounds):
    beta = ns.beta
    interval = 0
    t = 0
    event_times = [] 
    exceed_interval = False
    stop = False 

    while not stop:
        if not exceed_interval:
            u1 = np.random.uniform(0,1)
            x = -1 / upperbounds[interval] * np.log(u1)

        if (t + x) < time[interval + 1]:
            exceed_interval = False

            t += x
            u2 = np.random.uniform(0,1)

            lambda_t = ns.lambda_fun(t, u, v)
            prob = lambda_t / upperbounds[interval]

            assert lambda_t <= upperbounds[interval], "Lambda value exceeds upperbound"
            assert prob <= 1, "Probability out of range >1"
            assert prob >= 0, "Probability out of range <0"

            if u2 < prob: # accept event time t if u2 < lambda(t) / lambda upperbound
                event_times.append(t)

        elif (t + x) >=  time[interval + 1]:
            exceed_interval = True
            if interval == len(upperbounds) - 1:
                stop = True
            else:
                x = (x - time[interval + 1] + t) * upperbounds[interval] / upperbounds[interval+1]
                t = time[interval + 1]
                interval += 1

    return np.array(event_times)

# %%

def nhpp_mat(ns, time, root_matrix, monotonicity_matrix):
    beta = ns.beta
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    nhpp_mat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        r = root_matrix[u,v]
        m = monotonicity_matrix[u,v]

        # map time to monotonicity
        t2m = time_to_monotonicity(time, roots=r, monotonicity=m) 
        # find upperbounds for all time intervals
        ubl = upperbounds(ns, u, v, time, roots=r, time_to_monotonicity=t2m) 
        # simulate nhpp
        nhpp_sim = nhpp(ns, u, v, time=time, upperbounds=ubl) 

        nhpp_mat[u,v] = nhpp_sim
    
    return nhpp_mat



t = np.linspace(0, 10, 25)
nhppmat = nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

e = nhppmat[0,20]
plt.hist(e)
plt.show()

print(rmat[0,20])