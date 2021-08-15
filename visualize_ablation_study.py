import torch
import nodespace
#import smallnet_eucliddist as smallnet
import smallnet_node_specific_beta as smallnet
import os
import nhpp
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs

#MODEL_NAME = "CommonBeta_TrainAll"
#MODEL_NAME = "CommonBeta"
MODEL_NAME = "NewNewNodeSpecificBeta"
DATA_SET = "Radoslaw"

FILE_DIRECTORY = os.path.join(f"ablation_study\{DATA_SET+MODEL_NAME}", MODEL_NAME).replace(os.sep, '/')
MODEL_STATE_DICT = "ModelStateDict"
OPTIM_STATE_DICT = "OptimStateDict"
SEP = "_"
FILE_EXT = ".pth"
YLIM = (-4,10)
LEGEND_LOCATION = "upper left"

modes = ["b", "bz", "bzv", "bzva"]
#random_seeds  = [7, 8, 9]
random_seeds = [9]

seed = random_seeds[0]

train_loss_sfname = MODEL_NAME + SEP + "TrainLoss" + SEP + f"seed={seed}_mode=bzva" + FILE_EXT
test_loss_sfname = MODEL_NAME + SEP + "TestLoss" + SEP + f"seed={seed}_mode=bzva" + FILE_EXT

trainloss_list = torch.load(os.path.join(FILE_DIRECTORY, train_loss_sfname))
testloss_list = torch.load(os.path.join(FILE_DIRECTORY, test_loss_sfname))

def unpack(loss_list):
    loss_values = []
    for loss in loss_list:
        for value in loss:
            loss_values.append(value)
    return loss_values

train_loss = unpack(trainloss_list)
test_loss = unpack(testloss_list)

if "NodeSpecific" in MODEL_NAME:
    labels=[r"$\mathbf{\beta}$", r"$\mathbf{\beta}, \mathbf{Z}$", 
    r"$\mathbf{\beta}, \mathbf{Z}, \mathbf{V}$", 
    r"$\mathbf{\beta}, \mathbf{Z}, \mathbf{V}, \mathbf{A}$"]
else:
    labels=[r"$\beta$", r"$\beta, \mathbf{Z}$", 
    r"$\beta, \mathbf{Z}, \mathbf{V}$", 
    r"$\beta, \mathbf{Z}, \mathbf{V}, \mathbf{A}$"]

x_epochs = np.arange(1,(len(train_loss)+1))
# plt.plot(x_epochs,train_loss, color="blue", label="Train")
# plt.plot(x_epochs,test_loss, color="green", label="Test")
# plt.title(f"Loss progression by adding parameters {seed-(random_seeds[0]-1)}")
# #plt.title(f"Train all parameters")
# plt.vlines(x=[50,100,150, 200], ymin=-2.5, ymax=0.5, color="black", alpha=0.65)
# plt.ylabel("NLL")
# plt.xlabel("Included parameters")
# plt.xticks([25,75,125, 175], labels)
# plt.legend(loc="upper right")
# plt.show()

i, cmap = 1, plt.get_cmap("tab10")
for seed in random_seeds:
    train_loss_sfname = MODEL_NAME + SEP + "TrainLoss" + SEP + f"seed={seed}_mode=bzva" + FILE_EXT
    test_loss_sfname = MODEL_NAME + SEP + "TestLoss" + SEP + f"seed={seed}_mode=bzva" + FILE_EXT
    trainloss_list = torch.load(os.path.join(FILE_DIRECTORY, train_loss_sfname))
    testloss_list = torch.load(os.path.join(FILE_DIRECTORY, test_loss_sfname))
    train_loss = unpack(trainloss_list)
    test_loss = unpack(testloss_list)

    plt.plot(x_epochs, train_loss, color=cmap(i), label=f"Train {seed - (random_seeds[0]-1)}")
    plt.plot(x_epochs, test_loss, color=cmap(i+1), label=f"Test {seed - (random_seeds[0]-1)}")
    i+=2
plt.ylabel("NLL")
leg = plt.legend(loc=LEGEND_LOCATION, ncol=len(random_seeds), frameon=False)
leg.get_frame().set_linewidth(0.0)
plt.ylim(YLIM[0],YLIM[1])
plt.title(f"{DATA_SET+MODEL_NAME}")
if "TrainAll" in MODEL_NAME:
    plt.xticks([0,10,20,30,40,50])
    plt.vlines(x=[10,20,30,40,50], ymin=-2.5, ymax=0.0, color="black", alpha=0.65)
    plt.xlabel("Epoch")
else:
    plt.xticks([25,75,125, 175], labels)
    plt.xlabel("Included parameters")
    plt.vlines(x=[50,100,150, 200], ymin=-2.5, ymax=0.5, color="black", alpha=0.65)
plt.show()


#%%

seed = [9]
for s in seed:
    print(s)

# %%
