import torch
import nodespace
import smallnet_sqdist as smallnet
import os
import nhpp_mod
import compare_rates
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Set seeds
# Train 10 models with minibatch size = n_train
# Train 10 models with minibatch size = 70
# Save the models, save the losses

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

# Create dynamical system with constant velocity
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 7.5

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
n_points=len(z_gt)
ns_gt.init_conditions(z_gt, v_gt, a_gt)


# Simulate event time data set for the two nodes
t = np.linspace(0, 10)
rmat = nhpp_mod.root_matrix(ns_gt) 
mmat = nhpp_mod.monotonicity_mat(ns_gt, rmat)
nhppmat = nhpp_mod.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

# create data set and sort by time
ind = np.triu_indices(n_points, k=1)
data_set = []
for u,v in zip(ind[0], ind[1]):
    event_times = nhpp_mod.get_entry(nhppmat, u=u, v=v)
    for e in event_times:
        data_set.append([u, v, e])

data_set = np.array(data_set)
time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t


#np.save("4-point-data-set.npy", data_set)
verify_data = np.load("4-point-data-set.npy", allow_pickle=True)

print("Simulated: ", data_set[0])
print("Saved: ", verify_data[0])

data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
print("n_train:", n_train)
print("n_test:", len(test_data))

init_beta = smallnet.infer_beta(n_points, training_data)
print("init_beta:", init_beta)

training_batches = np.array_split(training_data, 450)

batch_size = len(training_batches[0])
print("Batch size:", batch_size)


z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
gt_net = smallnet.SmallNet(4, 7.5)
gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_net.load_state_dict(gt_dict)
gt_nt = gt_net.eval()

track_nodes=[0,3]
tn_train = training_batches[-1][-1][2] # last time point in training data
tn_test = test_data[-1][2] # last time point in test data

def getres(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_sq_fun(ti, track_nodes[0], track_nodes[1]))
    return torch.tensor(res)


def plotres(num_epochs, y_train, y_test, title):
    x = np.arange(num_epochs)
    plt.plot(x, y_train, "g", label="Train")
    plt.plot(x, y_test, "r", label="Test")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()
    plt.close()

def plotgrad(num_epochs, bgrad, zgrad, vgrad):
    x = np.arange(num_epochs)
    plt.plot(x, bgrad, "g", label="beta")
    plt.plot(x, zgrad, "b", label="z")
    plt.plot(x, vgrad, "r", label="v")
    plt.legend(loc="upper right")
    plt.title("Parameters: Mean abs grad for epoch")
    plt.show()
    plt.close()

res_gt=[getres(0, tn_train, gt_net, track_nodes), getres(tn_train, tn_test, gt_net, track_nodes)]    

NUM_EPOCHS = 11
NUM_INITS = 1
plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    #seed = ZERO_SEED + initialization
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)

    fpath = r"state_dicts/training_experiment"
    fname = f"batch={batch_size}_LR=0.001_reproduce_result_{seed}" + ".pth"
    full_path = os.path.join(fpath, fname)
    
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta)
    
    # if load previous
    #net.load_state_dict(torch.load(full_path))

    #net, train_loss, test_loss = smallnet.single_batch_train(net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    #net, train_loss, test_loss, track_dict = smallnet.single_batch_train_track_mse(res_gt, track_nodes, net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    #net, train_loss, test_loss = smallnet.batch_train(net, n_train, training_batches, test_data, NUM_EPOCHS)
    net, train_loss, test_loss, track_dict = smallnet.batch_train_track_mse(res_gt, track_nodes, net, n_train, training_batches, test_data, NUM_EPOCHS)

    plotres(NUM_EPOCHS, train_loss, test_loss, "LL Loss")
    plotres(NUM_EPOCHS, track_dict["mse_train_losses"], track_dict["mse_test_losses"], "MSE Loss")
    plotgrad(NUM_EPOCHS, track_dict["bgrad"], track_dict["zgrad"], track_dict["vgrad"])
    
    torch.save(net.state_dict(), full_path)

    compare_rates.compare_intensity_rates(net, ns_gt, 0, 3, full_path, training_data, test_data)
    
