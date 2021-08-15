import torch
import nodespace
import smallnet_eucliddist as smallnet
import os
import nhpp
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs

MODEL_NAME = "CommonBeta_TrainAll"
FILE_DIRECTORY = os.path.join("ablation_study", MODEL_NAME).replace(os.sep, '/')
MODEL_STATE_DICT = "ModelStateDict"
OPTIM_STATE_DICT = "OptimStateDict"
SEP = "_"
FILE_EXT = ".pth"


n_points=150
data_set = np.load("150-point-data-set-b=3.55-t=10.npy", allow_pickle=True)
print("n_events:", len(data_set))

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

n_train = len(training_data)
n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)


NUM_SPLITS = 550
training_batches = np.array_split(training_data, NUM_SPLITS)
batch_size = len(training_batches[0])
print("Batch size:", batch_size)


nps = batch_size * 5
init_beta = torch.nn.init.uniform_(torch.zeros(1,1), a=0.0, b=0.025)
rs = 1
niw = 1
num_epochs = 50

#%%
modes = ["b", "bz", "bzv", "bzva"]
FIXED_MODE = 3
random_seeds  = [7, 8, 9]

# Train all parameters immediately on three initializations

for seed in random_seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("seed", seed)
    seed_train_loss = []
    seed_test_loss = []

    model = smallnet.SmallNet(FIXED_MODE, n_points, init_beta, rs, nps, niw)
    model, train_loss, test_loss, optimizer = smallnet.batch_train_scheduled(model, n_train, training_data, training_batches, test_data, num_epochs)

    seed_train_loss.append(train_loss)
    seed_test_loss.append(test_loss)
    # append train and test loss to list
    
    # save current model
    model_sfname = MODEL_NAME + SEP + MODEL_STATE_DICT + SEP + f"seed={seed}_mode={modes[FIXED_MODE]}" + FILE_EXT
    torch.save(model.state_dict(), os.path.join(FILE_DIRECTORY,model_sfname))

    # save optimizer state dict
    optim_sfname = MODEL_NAME + SEP + OPTIM_STATE_DICT + SEP + f"seed={seed}_mode={modes[FIXED_MODE]}" + FILE_EXT
    torch.save(optimizer.state_dict(), os.path.join(FILE_DIRECTORY,optim_sfname))

    # save train loss for seed
    train_loss_sfname = MODEL_NAME + SEP + "TrainLoss" + SEP + f"seed={seed}_mode={modes[FIXED_MODE]}" + FILE_EXT
    torch.save(seed_train_loss, os.path.join(FILE_DIRECTORY, train_loss_sfname))

    # save test loss for seed
    test_loss_sfname = MODEL_NAME + SEP + "TestLoss" + SEP + f"seed={seed}_mode={modes[FIXED_MODE]}" + FILE_EXT
    torch.save(seed_test_loss, os.path.join(FILE_DIRECTORY, test_loss_sfname))