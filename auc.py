import torch
import nodespace
import smallnet_eucliddist as smallnet # CHOOSE CORRECT MODEL 
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

fpath = r"state_dicts/training_experiment"
fname = f"radoslaw-bs148-gc30-smpl=1-train-all-lrschedul" + ".pth"
full_path = os.path.join(fpath, fname)

n_points=167
init_beta = 0.0
node_pair_samples=750
model = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=1, node_pair_samples=node_pair_samples, non_intensity_weight=1)

model.load_state_dict(torch.load(full_path))
model.eval()


data_set = np.load("radoslaw-email.npy", allow_pickle=True)

num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

start_t = test_data[0][2]
end_t = test_data[-1][2]
total_node_pairs = n_points*(n_points-1)/2

triu_indices = torch.triu_indices(row=n_points,col=n_points, offset=1)

#%%
bins = np.linspace(start_t,end_t, 10) 

#%%
t0 = bins[1]
tn = bins[2]

test_data = torch.from_numpy(test_data)
binned_data = test_data[torch.logical_and(test_data[:,2]>=t0, test_data[:,2]<=tn)]

pos_node_pairs = torch.unique(binned_data[:,:-1], dim=0).to(dtype=torch.int64)
print("Positives:", len(pos_node_pairs))
#%%
# map to index
pos_indices = torch.tensor([to_triu_idx(n_points, node_pair) for node_pair in pos_node_pairs])

indices = torch.arange(0, total_node_pairs)
neg_indices = torch.tensor([idx for idx in indices if not idx in pos_indices])


# Draw n_pos random negatives without replacement
n_pos = len(pos_indices)
permuted_idx = torch.randperm(neg_indices.nelement())
neg_indices = neg_indices.view(-1)[permuted_idx].view(neg_indices.size()).to(dtype=torch.int64)
neg_indices = neg_indices[:n_pos]

#%%

pos_score = predict(model, pos_indices, t0, tn, triu_indices)
neg_score = predict(model, neg_indices, t0, tn, triu_indices)

ytrue = true_labels(pos_indices, neg_indices)
yscore = concat_score(pos_score, neg_score)

auc = sm.roc_auc_score(y_true=ytrue, y_score=yscore)
print("AUC score:", auc)