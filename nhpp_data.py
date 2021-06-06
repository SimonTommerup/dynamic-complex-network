import nhpp_mod
import time
import cmath
import numpy as np
import scipy as sp
from nodespace import NodeSpace

# initialize system
ns = NodeSpace()
n_clusts = 2
points_in_clusts = [1, 1]
n_points = sum(points_in_clusts)
centers = [[-6,0], [6,1]]
radius = [0,0]
v = [[1,0], [-1,0]]
a =  [[0,0], [0,0]]
z0 = ns.init_clusters(n_clusts, points_in_clusts, centers, radius)
v0, a0 = ns.init_dynamics(n_clusts, points_in_clusts, v, a)
ns.init_conditions(z0, v0, a0)

np.seterr(all='raise')

# beta and alpha: lambda = exp(beta - alpha * dist)
ns.beta = 7.5
ns.alpha = 0.1

# time 
t = np.linspace(0, 10)

print("Simulating events...")
t0 = time.time()
rmat = nhpp_mod.root_matrix(ns) 

print(nhpp_mod.get_entry(rmat, 0,1))


mmat = nhpp_mod.monotonicity_mat(ns, rmat)
nhppmat = nhpp_mod.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
print("Elapsed simulation time (s): ", time.time() - t0)

# %%-
dataset = nhppmat[0][1]
print("No events: ", len(dataset))


# %%
import scipy.special as sps
sps.erf(1)

def llint(z, v, T, alpha, beta):
    a = z[0,0] - z[1,0]
    b = z[0,1] - z[1,1]
    m = v[0,0] - v[1,0]
    n = v[0,1] - v[1,1]
    return np.sqrt(np.pi)*np.exp((-(a*n - b*m)**2*alpha + beta*(m**2 + n**2))/(m**2 + n**2))*np.sqrt(alpha*(m**2 + n**2))*(sps.erf(alpha*((m**2 + n**2)*T + a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))) - sps.erf(alpha*(a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))))/(2*alpha*(m**2 + n**2))


# %%
# Exact integral:
exact = llint(ns.z0, ns.v0, t[-1], ns.alpha, ns.beta)
# Compare to Riemann approximation
approx = ns.lambda_int_sq_rapprox(t, u=0, v=1)

print("Exact:", exact)
print("Approx: ", approx)
# %%

# Compare likelihood of parameter estimates

# likelihood for two nodes
def loglikelihood(ns, event_time, T, weight=1):
    print("Position:")
    print(ns.z0)
    event_intensity = 0.
    for t in event_time:
        event_intensity += ns.lambda_sq_fun(t, 0, 1)
    
    non_event_intensity = llint(ns.z0, ns.v0, T, ns.alpha, ns.beta)
    
    print("Event intensity:")
    print(event_intensity)
    print("Nonevent intensity:")
    print(non_event_intensity)

    return event_intensity - weight*non_event_intensity

#%%
# Groundtruth
# centers = [[-6,0], [6,1]]
# v = [[1,0], [-1,0]]
# a =  [[0,0], [0,0]]

# estimated system:
ns_est = NodeSpace()
ns_est.beta = 7.5
ns_est.alpha = 0.1
z_est = np.array([[2.,3.], [-3.,2.]])
v_est = np.array([[5.,0.], [-5.,0.]])
a_est = np.array([[0.,0.], [0.,0.]])
ns_est.init_conditions(z_est, v_est, a_est)

# %%
# Likelihood ground truth
gt_ll = loglikelihood(ns, dataset, t[-1])
print("Groundtruth:", gt_ll)

#%%
# likelihood estimate
est_ll = loglikelihood(ns_est, dataset, t[-1])
print("Estimate:", est_ll)
print("Worse than gt:", est_ll < gt_ll)

#%%

# %%
# estimated system "improved":
# Groundtruth
# centers = [[-6,0], [6,1]]
# v = [[1,0], [-1,0]]
# a =  [[0,0], [0,0]]
ns_est_2 = NodeSpace()
ns_est_2.beta = 7.5
ns_est_2.alpha = 0.1
z_est_2 = np.array([[-6, 1], [6.,1.]])
v_est_2 = np.array([[1,0.], [-1,0.]])
a_est_2 = np.array([[0.,0.], [0.,0.]])
ns_est_2.init_conditions(z_est_2, v_est_2, a_est_2)

# likelihood estimate
weight = 1.25e2
est_ll_2 = loglikelihood(ns_est_2, dataset, t[-1], weight=weight)

# estimate dist
#print("Dist", ns_est_2.get_sq_dist(0, 0, 1))

print("GT:", gt_ll)
print("Estimate:", est_ll_2)

print("Better than GT:", est_ll_2 > gt_ll)
print("Better than first est:", est_ll_2 > est_ll)
# %%



