import torch
import nodespace
#import smallnet_eucliddist as smallnet # CHOOSE CORRECT MODEL
import smallnet_node_specific_beta as smallnet # CHOOSE CORRECT MODEL 
import os
import nhpp
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sklearn.metrics as sm
from sklearn.preprocessing import label_binarize


def to_triu_idx(n_points, node_pair):
  u, v = node_pair[0], node_pair[1]
  if u > v:
      u,v = v,u
  triu_idx = int(n_points*u+v - ((u+2) * (u+1)) / 2)
  return triu_idx

def true_labels(pos_indices, neg_indices):
  positive_labels = np.ones(shape=(len(pos_indices),))
  negative_labels = np.zeros(shape=(len(neg_indices),))
  labels = np.concatenate([positive_labels, negative_labels], axis=0)
  return labels

def predict(model, sample_indices, t0, tn, triu_indices):
  scores = []
  for idx in sample_indices:
    u, v = triu_indices[0][idx], triu_indices[1][idx]
    score = model.riemann_sum(u, v, t0, tn, n_samples=10).item()
    scores.append(score)
  return np.array(scores)

def concat_score(pos_score, neg_score):
  arr = np.concatenate([pos_score, neg_score], axis=0)
  return arr

def burstiness(data):
  t, n = data[:,2], len(data[:,2])
  inter_arrivals = np.diff(t)
  mu, sigma = np.mean(inter_arrivals), np.std(inter_arrivals)
  b = (sigma - mu)/(sigma + mu)
  return b


fpath = r"state_dicts/training_experiment"
#fname = f"radoslaw-bs148-gc30-smpl=1-train-all-lrschedul" + ".pth"
#fname = f"radoslaw-bs148-nps=750-gc30-smpl=1-node-specific-beta" + ".pth"
#fname = f"synthetic-150-bs=148-gc=30-train-all" + ".pth"
#fname = f"synthetic-150-bs=148-gc=30-pretrain-beta-zva" + ".pth"
fname = f"synthetic-150-bs=139-nps=695-gc=30-train-all-nodespecificbeta" + ".pth"
#fname = f"hospital-bs146-gc30-rs=1-nps=73-train-all-scheduled" + ".pth"
#fname = f"contact-bs146-gc30-rs=1-nps=73-train-all-scheduled" + ".pth"
#fname = f"contact-bs113-gc30-rs=1-nps=565-train-all-scheduled-2" + ".pth"
#fname = r"hospital-bs146-gc30-rs=1-nps=73-train-all-scheduled-newsplit80-20" + ".pth"
full_path = os.path.join(fpath, fname)


#n_points=167  # radoslaw
n_points=150  # synthetic
#n_points = 75 # hospital
#n_points = 274  # contact
init_beta = 0.0
#node_pair_samples=740 # radoslaw
#node_pair_samples=740 # synthetic#
#node_pair_samples=730 # hospital
node_pair_samples=695
model = smallnet.SmallNet(3, n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)

model.load_state_dict(torch.load(full_path))
model.eval()


#data_set = np.load("radoslaw-email.npy", allow_pickle=True)
#data_set = np.load("hospital-proximity.npy", allow_pickle=True)
#data_set = np.load("contact.npy", allow_pickle=True)
data_set = np.load("150-point-data-set-b=3.55-t=10.npy", allow_pickle=True)

num_train_samples = int(len(data_set)*0.8) #radoslaw
#num_train_samples = int(len(data_set)*0.85) #synthetic-150
#num_train_samples = int(len(data_set)*0.9) #hospital-proximity
#num_train_samples = int(len(data_set)*0.9) # contact
#num_train_samples = int(len(data_set)*0.8) # contact-2
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]


print("Burstiness:", burstiness(data_set))
# synthetic = 0.12
# radoslaw = 0.87
# hospital = 0.89
# contact = 0.77


start_t = test_data[0][2]
end_t = test_data[-1][2]
total_node_pairs = n_points*(n_points-1)/2

triu_indices = torch.triu_indices(row=n_points,col=n_points, offset=1)

#%%
bins = np.linspace(start_t,end_t, 10) 

#%%
t0 = bins[4]
tn = bins[5]

test_data = torch.from_numpy(test_data)
binned_data = test_data[torch.logical_and(test_data[:,2]>=t0, test_data[:,2]<=tn)]

pos_node_pairs = torch.unique(binned_data[:,:-1], dim=0).to(dtype=torch.int64)
print("Positives:", len(pos_node_pairs))
#%%
# map to index
pos_indices = torch.tensor([to_triu_idx(n_points, node_pair) for node_pair in pos_node_pairs])

# Draw n_pos random negatives without replacement
indices = torch.arange(0, total_node_pairs)
neg_indices = torch.tensor([idx for idx in indices if not idx in pos_indices])
n_pos = len(pos_indices)
permuted_idx = torch.randperm(neg_indices.nelement())
neg_indices = neg_indices.view(-1)[permuted_idx].view(neg_indices.size()).to(dtype=torch.int64)
neg_indices = neg_indices[:n_pos]


#%%

import numpy as np
import matplotlib.pyplot as plt

nmax=0.001
nmin=0
tmult = 1
ti = 5
tcur=0
lr=[]
for i in range(20):
  if tcur == 0:
    lr.append(nmax)
    tcur+=1
  elif tcur == ti:
    lr.append(nmin)
    tcur=0
  else:
    lr.append(0.5*nmax*(1+np.cos(np.pi*tcur/ti)))
    tcur+=1

plt.plot([i for i in range(20)], lr)
plt.ylim(0,0.001)
plt.xticks([n for n in range(20)])
plt.show()


# %%
a=[-4165, -5515, -6252,-5938,-5323]
b=[4887,-3296,-8257,-6329,-39]
c=[5320, -3971, -8120, -6075, -215]
d=[2714,-4656, -7895, -5872, -3044]

print("b", sum(d))
# %%
