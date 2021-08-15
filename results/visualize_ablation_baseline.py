import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sklearn.metrics as sm


MODEL_NAME = "CommonBeta"
#MODEL_NAME = "NodeSpecificBeta"
DATA_SET = "Radoslaw"
FILE_DIRECTORY = os.path.join(f"ablation_study\{DATA_SET+MODEL_NAME}", MODEL_NAME).replace(os.sep, '/')
MODEL_STATE_DICT = "ModelStateDict"
OPTIM_STATE_DICT = "OptimStateDict"
MODE = "bzva"
SEP = "_"
FILE_EXT = ".pth"
DATA_FILE_EXT = ".npy"
POSSIBLE_MODES = ["b", "bz", "bzv", "bzva"]
RANDOM_SEEDS = [7,8,9]
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

def get_interacting_node_pairs(training_data, time_col):
    """
    Get seen links during training
    """
    if time_col:
        unique_links = torch.unique(training_data[:,:-1], dim=0)
    else:
        unique_links = torch.unique(training_data[:,:], dim=0)
    return unique_links

def swapcols(a):
    l = []
    for row in a:
        if row[0] > row[1]:
            l.append([row[1], row[0]])
        else:
            l.append([row[0], row[1]])
    return torch.tensor(l)

def get_score_dict(interacting_node_pairs, data_set):
    sd={}
    for np in interacting_node_pairs:
        u, v = np[0], np[1]
        count=0
        for row in training_data:
            if row[0]==u and row[1]==v:
                count += 1
            elif row[0]==v and row[1]==u:
                count += 1
            else:
                continue
        sd[np] = count
    return sd

def to_triu_idx(n_points, node_pair):
  u, v = node_pair[0], node_pair[1]
  if u > v:
      u,v = v,u
  triu_idx = int(n_points*u+v - ((u+2) * (u+1)) / 2)
  return triu_idx

data_set_fname = get_data_set_fname(DATA_SET, DATA_FILE_EXT)
data_set = np.load(data_set_fname, allow_pickle=True)

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


training_data = training_data[150:160]
print(training_data)

_nps = get_interacting_node_pairs(training_data, time_col=True)
#_nps = swapcols(_nps)
#interacting_node_pairs = get_interacting_node_pairs(_nps, time_col=False)

sd = get_score_dict(_nps, training_data)

print(sd)

#%%







# %%
