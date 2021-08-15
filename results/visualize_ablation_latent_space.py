import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)

MODEL_NAME = "CommonBeta"
#MODEL_NAME = "NodeSpecificBeta"
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
TIME_STEPS = [0, 3, 6, 9]
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

def floatcount(indices, data_set):
    counts = []
    for idx in indices:
        bin_data = data_set[np.logical_or(data_set[:,0]==idx, data_set[:,1]==idx)]
        count = [idx, len(bin_data)]
        counts.append(count)
    return np.array(counts)

def descending_degree(edge_counts, k):
    return np.argsort(edge_counts[:,1])[::-1][:k]

def ascending_degree(edge_counts,k):
    return np.argsort(edge_counts[:,1])[:k]

def paircount(u, v, data_set):
    uv_cond = np.logical_or(data_set[:,0]==u, data_set[:,1]==v)
    vu_cond = np.logical_or(data_set[:,0]==v, data_set[:,1]==u)
    bin_data = data_set[np.logical_or(uv_cond, vu_cond)]
    count = len(bin_data)
    return count

data_set_fname = get_data_set_fname(DATA_SET, DATA_FILE_EXT)
data_set = np.load(data_set_fname, allow_pickle=True)

# verify time ordering
prev_t = 0.
time_col=2
for row in data_set:
    cur_t = row[time_col]
    assert cur_t >= prev_t
    prev_t = cur_t

#data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]
binned_test_data = np.array_split(test_data, N_BINS)

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


n_points =n_points_dict[DATA_SET]
num_splits = split_dict[DATA_SET]

training_batches = np.array_split(training_data, num_splits)
batch_size = len(training_batches[0])
print(batch_size)

mode_idx = get_mode_idx(MODE, POSSIBLE_MODES)
init_beta = 0.0
rs = 1
nps = batch_size * 5
niw=1

seed=7
model_lfname = MODEL_NAME + SEP + MODEL_STATE_DICT + SEP + f"seed={seed}_mode={MODE}" + FILE_EXT
model = smallnet.SmallNet(mode_idx, n_points, init_beta, rs, nps, niw)
model.load_state_dict(torch.load(os.path.join(FILE_DIRECTORY, model_lfname)))
model.eval()

# Find most and least connected nodes in training data
indices = np.arange(n_points)
edge_counts = floatcount(indices, training_data)
top_nodes = descending_degree(edge_counts, 20)
low_nodes = ascending_degree(edge_counts,20)


NROWS, NCOLS = 2,2
fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=True, figsize=(8,8))
fig.suptitle(f"{DATA_SET} data:" + " " + f"{MODEL_NAME} model")
TIME_IDX = 0
for row in range(NROWS):
    for col in range(NCOLS):
        cur_z = model.step(TIME_STEPS[TIME_IDX]).detach().numpy()
        ax[row][col].scatter(cur_z[:,0], cur_z[:,1], color="black", label="All")
        ax[row][col].scatter(cur_z[top_nodes,0], cur_z[top_nodes,1], color="blue", label="Most connected")
        ax[row][col].scatter(cur_z[low_nodes,0], cur_z[low_nodes,1], color="red", label="Least connected")
        
        ax[row][col].set_title(f"t={TIME_STEPS[TIME_IDX]}")
        ax[row][col].grid()
        TIME_IDX += 1
fig.tight_layout()
#fig.show()



