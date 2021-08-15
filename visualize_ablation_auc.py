import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sklearn.metrics as sm

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)

#MODEL_NAME = "CommonBeta"
MODEL_NAME = "NodeSpecificBeta"
DATA_SET = "Hospital"
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

assert MODE in POSSIBLE_MODES, "No such MODE"

if MODEL_NAME == "CommonBeta":
    import smallnet_eucliddist as smallnet
elif MODEL_NAME == "NodeSpecificBeta":
    import smallnet_node_specific_beta as smallnet

def get_mode_idx(MODE, POSSIBLE_MODES):
    mode_idx = None
    for idx, mode in enumerate(POSSIBLE_MODES):
        if MODE == mode:
            mode_idx = idx
    return mode_idx

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


def to_triu_idx(n_points, node_pair):
  u, v = node_pair[0], node_pair[1]
  if u > v:
      u, v = v, u
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

#binned_test_data = np.array_split(test_data, N_BINS)

n_points_dict = {
    "Synthetic" : 150, 
    "Contact" : 274,
    "Hospital" : 75,
    "Radoslaw" : 167} 

split_dict = {
    "Synthetic" : 575, 
    "Contact" : 170,
    "Hospital" : 200,
    "Radoslaw" : 550} 

n_points = n_points_dict[DATA_SET]
num_splits = split_dict[DATA_SET]

training_batches = np.array_split(training_data, num_splits)
batch_size = len(training_batches[0])

mode_idx = get_mode_idx(MODE, POSSIBLE_MODES)
init_beta = 0.0
rs = 1
nps = batch_size * 5
niw=1


# AUC related settings
START_T = test_data[0][2]
END_T = test_data[-1][2]
total_node_pairs = n_points*(n_points-1)/2
time_bins = np.linspace(START_T,END_T, N_BINS+1)
triu_indices = torch.triu_indices(row=n_points,col=n_points, offset=1)

mode_prediction = np.zeros(shape=(3, N_BINS))
for seed_idx, seed in enumerate(RANDOM_SEEDS):
    print("SEED:", seed_idx + 1)
    model_lfname = MODEL_NAME + SEP + MODEL_STATE_DICT + SEP + f"seed={seed}_mode={MODE}" + FILE_EXT
    model = smallnet.SmallNet(mode_idx, n_points, init_beta, rs, nps, niw)
    model.load_state_dict(torch.load(os.path.join(FILE_DIRECTORY, model_lfname)))
    model.eval()

    with torch.no_grad():
        auc_scores = []
        for bin_idx in range(N_BINS):
            #print(f"{bin_idx} to {bin_idx+1}")
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
            pos_score = predict(model, pos_indices, bin_t0, bin_tn, triu_indices)
            neg_score = predict(model, neg_indices, bin_t0, bin_tn, triu_indices)

            ytrue = true_labels(pos_indices, neg_indices)
            yscore = concat_score(pos_score, neg_score)

            bin_auc = sm.roc_auc_score(y_true=ytrue, y_score=yscore)
            auc_scores.append(bin_auc)

    auc_scores = np.array(auc_scores)
    mode_prediction[seed_idx] = auc_scores

mu_prediction = np.mean(mode_prediction, axis=0)
sd_prediction = np.std(mode_prediction, axis=0)

print("Model mode:")
print(DATA_SET + MODEL_NAME)
print(MODE)
#print("Mean AUC for bins:")
#print(mu_prediction)
#print("Standard deviation AUC for bins:")
#print(sd_prediction)

row_tex_string=""
for idx, (mp, sd) in enumerate(zip(mu_prediction, sd_prediction)):
    if idx < (len(sd_prediction)-1):
        row_tex_string += "$" + str(round(mp,2)) + " \pm " + str(round(sd,2)) + "$ & "
    else:
        row_tex_string += "$" + str(round(mp,2)) + " \pm " + str(round(sd,2)) + "$ \\\\"

print(row_tex_string)
