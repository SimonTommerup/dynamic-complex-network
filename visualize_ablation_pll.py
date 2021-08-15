import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

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

mode_idx = get_mode_idx(MODE, POSSIBLE_MODES)
init_beta = 0.0
rs = 1
nps = batch_size * 5
niw=1

mode_prediction = np.zeros(shape=(3, N_BINS))
for seed_idx, seed in enumerate(RANDOM_SEEDS):
    model_lfname = MODEL_NAME + SEP + MODEL_STATE_DICT + SEP + f"seed={seed}_mode={MODE}" + FILE_EXT
    model = smallnet.SmallNet(mode_idx, n_points, init_beta, rs, nps, niw)
    model.load_state_dict(torch.load(os.path.join(FILE_DIRECTORY, model_lfname)))
    model.eval()
    with torch.no_grad():
        pnll = []
        for time_bin in range(N_BINS):
            bin_data = binned_test_data[time_bin]
            bin_t0, bin_tn = bin_data[0][2], bin_data[-1][2]
            
            ll_pred = model(bin_data, bin_t0, bin_tn)
            nll_pred = smallnet.nll(ll_pred)

            pnll.append(nll_pred)
        
    pnll = np.array(pnll)
    mode_prediction[seed_idx] = pnll

mu_prediction = np.mean(mode_prediction, axis=0)
sd_prediction = np.std(mode_prediction, axis=0)

print("Model mode:")
print(DATA_SET + MODEL_NAME)
print(MODE)
#print("Mean prediction for bins:")
print("TABLE ROW:")
# mp_tex_string=""
# for idx, mp in enumerate(mu_prediction):
#     if idx < (len(mu_prediction)-1):
#         mp_tex_string += str(mp) + " " + "& "
#     else:
#         mp_tex_string += str(mp) + " \\\\"

# #print(mp_tex_string)

# #print("Standard deviation prediction for bins:")
# sd_tex_string=""
# for idx, mp in enumerate(sd_prediction):
#     if idx < (len(sd_prediction)-1):
#         sd_tex_string += str(mp) + " " + "& "
#     else:
#         sd_tex_string += str(mp) + " \\\\"

# #print(sd_tex_string)

row_tex_string=""
for idx, (mp, sd) in enumerate(zip(mu_prediction, sd_prediction)):
    if idx < (len(sd_prediction)-1):
        row_tex_string += "$" + str(round(mp,2)) + " \pm " + str(round(sd,2)) + "$ & "
    else:
        row_tex_string += "$" + str(round(mp,2)) + " \pm " + str(round(sd,2)) + "$ \\\\"

print(row_tex_string)