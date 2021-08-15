import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sklearn.metrics as sm

DATA_SET = "Hospital"
DATA_FILE_EXT = ".npy"
N_BINS = 5

def get_data_set_fname(DATA_SET, DATA_FILE_EXT):
    assert DATA_SET in ["Synthetic", "Radoslaw", "Contact", "Hospital"], "No such DATA_SET."
    DATA_FILE_NAME=""
    if DATA_SET=="Synthetic":
        DATA_FILE_NAME="150-point-data-set-b=3.55-t=10" + DATA_FILE_EXT
    elif DATA_SET=="Radoslaw":
        DATA_FILE_NAME="radoslaw-email" + DATA_FILE_EXT
    elif DATA_SET=="Contact":
        DATA_FILE_NAME = "contact" + DATA_FILE_EXT
    else:
        DATA_FILE_NAME = "hospital-proximity" + DATA_FILE_EXT
    return DATA_FILE_NAME

data_set_fname = get_data_set_fname(DATA_SET, DATA_FILE_EXT)
data_set = np.load(data_set_fname, allow_pickle=True)

def to_triu_idx(n_points, node_pair):
  u, v = node_pair[0], node_pair[1]
  if u > v:
      u,v = v,u
  triu_idx = int(n_points*u+v - ((u+2) * (u+1)) / 2)
  return triu_idx

n_points_dict = {
    "Synthetic" : 150, 
    "Contact" : 274,
    "Hospital" : 75,
    "Radoslaw" : 167} 

n_points = n_points_dict[DATA_SET]

# verify time ordering
prev_t = 0.
time_col=2
for row in data_set:
    cur_t = row[time_col]
    assert cur_t >= prev_t
    prev_t = cur_t

data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

# Go through training data and count each occurence of a node pair
score_dict = {}
for row in training_data:
    u, v = row[0], row[1]
    node_pair_idx = to_triu_idx(n_points, [u,v])
    try:
        score_dict[node_pair_idx] += 1
    except KeyError:
        score_dict[node_pair_idx] = 1


# baseline prediction function
def baseline_prediction(score_dict, indices):
    scores = []
    for idx in indices:
        idx = idx.item()
        try:
            score = score_dict[idx]
        except KeyError:
            score = 0
        scores.append(score)
    return np.array(scores)

def true_labels(pos_indices, neg_indices):
  positive_labels = np.ones(shape=(len(pos_indices),))
  negative_labels = np.zeros(shape=(len(neg_indices),))
  labels = np.concatenate([positive_labels, negative_labels], axis=0)
  return labels

def concat_score(pos_score, neg_score):
  arr = np.concatenate([pos_score, neg_score], axis=0)
  return arr

# AUC related settings
START_T = test_data[0][2]
END_T = test_data[-1][2]
total_node_pairs = n_points*(n_points-1)/2
time_bins = np.linspace(START_T,END_T, N_BINS+1)
triu_indices = torch.triu_indices(row=n_points,col=n_points, offset=1)


auc_scores = []
for bin_idx in range(N_BINS):
    print(f"{bin_idx} to {bin_idx+1}")
    bin_t0, bin_tn = time_bins[bin_idx], time_bins[bin_idx+1]
    binned_data = test_data[torch.logical_and(test_data[:,2]>=bin_t0, test_data[:,2]<=bin_tn)]
    
    # find positives in bin
    pos_node_pairs = torch.unique(binned_data[:,:-1], dim=0).to(dtype=torch.int64)
    pos_indices = torch.tensor([to_triu_idx(n_points, node_pair) for node_pair in pos_node_pairs])
    n_pos = len(pos_indices)

    # draw n_pos random negatives without replacement
    indices = torch.arange(0, total_node_pairs)
    neg_indices = torch.tensor([idx for idx in indices if not idx in pos_indices])
    permuted_idx = torch.randperm(neg_indices.nelement())
    neg_indices = neg_indices.view(-1)[permuted_idx].view(neg_indices.size()).to(dtype=torch.int64)
    neg_indices = neg_indices[:n_pos]

    # calculate score for bin
    pos_score = baseline_prediction(score_dict, pos_indices)
    neg_score = baseline_prediction(score_dict, neg_indices)

    ytrue = true_labels(pos_indices, neg_indices)
    yscore = concat_score(pos_score, neg_score)

    bin_auc = sm.roc_auc_score(y_true=ytrue, y_score=yscore)
    auc_scores.append(bin_auc)

auc_scores = np.array(auc_scores)
#print("AUC for bins:")
#print(auc_scores)

print(DATA_SET)
row_tex_string=""
for idx, s in enumerate(auc_scores):
    if idx < (len(auc_scores)-1):
        row_tex_string += "$" + str(round(s,2)) +  "$ & "
    else:
        row_tex_string += "$" + str(round(s,2)) + "$ \\\\"

print(row_tex_string)
#%%

# %%

