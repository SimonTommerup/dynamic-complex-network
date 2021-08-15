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

ZERO_SEED = 0
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

# Create dynamical system with constant velocity
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 7.5

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.001,0.002], [-0.002,-0.001], [0.002,-0.002], [-0.002, 0.001]])
n_points=len(z_gt)
ns_gt.init_conditions(z_gt, v_gt, a_gt)


# Simulate event time data set for the two nodes
t = np.linspace(0, 15)
rmat = nhpp.root_matrix(ns_gt) 
mmat = nhpp.monotonicity_mat(ns_gt, rmat)
nhppmat = nhpp.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

# create data set and sort by time
ind = np.triu_indices(n_points, k=1)
data_set = []
for u,v in zip(ind[0], ind[1]):
    event_times = nhpp.get_entry(nhppmat, u=u, v=v)
    for e in event_times:
        data_set.append([u, v, e])

data_set = np.array(data_set)
time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]

plt.hist(data_set[:,2])
plt.show()
plt.close()

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t > prev_t
    prev_t = cur_t


#np.save("4-point-acceleration-data-set.npy", data_set)
verify_data = np.load("4-point-acceleration-data-set.npy", allow_pickle=True)

print("Simulated: ", data_set[0])
print("Saved: ", verify_data[0])

data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
n_test = len(test_data)
print("n_train:", n_train)
print("n_test:", n_test)

#%%

#init_beta = torch.nn.init.uniform_(torch.zeros(size=(1,1)), a=0.0, b=1.0)
init_beta = smallnet.infer_beta(4, training_data)
print("init_beta:", init_beta)

training_batches = np.array_split(training_data, 500) 
# split=1600 => bs=45
# split=500 => bs=141

batch_size = len(training_batches[0])
print("Batch size:", batch_size)


z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.001,0.002], [-0.002,-0.001], [0.002,-0.002], [-0.002, 0.001]])
gt_net = smallnet.SmallNet(n_points=n_points, init_beta=0.0, riemann_samples=1, node_pair_samples=6, non_intensity_weight=1)
gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_a = torch.from_numpy(a_gt)
gt_dict["beta"] = torch.from_numpy(np.array([[ns_gt.beta]]))
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_dict["a0"] = gt_a
gt_net.load_state_dict(gt_dict)
gt_nt = gt_net.eval()

track_nodes=[0,1]
tn_train = training_batches[-1][-1][2] # last time point in training data
tn_test = test_data[-1][2] # last time point in test data


# gt_output_train, _ = gt_net(training_data, t0=torch.zeros(size=(1,1)), tn=tn_train)
# gt_output_test, _ = gt_net(test_data, t0=tn_train, tn=tn_test)

# gt_loglik_train = smallnet.nll(gt_output_train) / n_train
# gt_loglik_test = smallnet.nll(gt_output_test) / n_test

# print("GT:")
# print("Train likelihood:", gt_loglik_train.item())
# print("Test likelihood:", gt_loglik_test.item())


def getres(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_fun(ti, track_nodes[0], track_nodes[1]))
    return torch.tensor(res)


def plotres(num_epochs, y_train, y_test, title):
    x = np.arange(num_epochs)
    plt.plot(x, y_train, "g", label="Train")
    plt.plot(x, y_test, "r", label="Test")
    plt.legend(loc="upper right")
    plt.ylabel(title)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.show()
    plt.close()

def plotgrad(num_epochs, bgrad, zgrad, vgrad, agrad):
    x = np.arange(num_epochs)
    plt.plot(x, bgrad, "g", label="beta")
    plt.plot(x, zgrad, "b", label="z")
    plt.plot(x, vgrad, "r", label="v")
    plt.plot(x, agrad, "m", label="a")
    plt.legend(loc="upper right")
    plt.title("Parameters: Mean abs grad for epoch")
    plt.ylabel("mean abs value")
    plt.xlabel("Epoch")
    plt.show()
    plt.close()

res_gt=[getres(0, tn_train, gt_net, track_nodes), getres(tn_train, tn_test, gt_net, track_nodes)]    

NUM_EPOCHS = 20
NUM_INITS = 1
plt.ion()
for initialization in range(1,NUM_INITS + 1):
    print(f"Initialization {initialization}")
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)

    fpath = r"state_dicts/training_experiment"
    load_fname = f"batch=141-epoch=15-gradclip=30-riemann=5-std=0.025-pretrain-bzv-init=2"
    ext = ".pth"
    full_path = os.path.join(fpath, load_fname + ext)
    
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=5, node_pair_samples=6)
    
    # if load previous
    net.load_state_dict(torch.load(full_path))

    net.beta.requires_grad = False
    net.z0.requires_grad = False
    net.v0.requires_grad = False
    net.a0.requires_grad = True

    statedict = net.state_dict()
    statedict["a0"] = torch.nn.init.normal_(torch.zeros(size=(n_points,2)), mean=0.0, std=0.001)
    net.load_state_dict(statedict)

    #net, train_loss, test_loss = smallnet.single_batch_train(net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    #net, train_loss, test_loss, track_dict = smallnet.single_batch_train_track_mse(res_gt, track_nodes, net, num_train_samples, training_data, test_data, NUM_EPOCHS)
    #net, train_loss, test_loss = smallnet.batch_train(net, n_train, training_data, training_batches, test_data, NUM_EPOCHS)

    net, train_loss, test_loss, track_dict = smallnet.batch_train_track_mse(res_gt, track_nodes, net, n_train, training_batches, test_data, NUM_EPOCHS)

    plotres(NUM_EPOCHS, train_loss, test_loss, "NLL Loss")
    plotres(NUM_EPOCHS, track_dict["mse_train_losses"], track_dict["mse_test_losses"], "MSE Loss")
    #plotgrad(NUM_EPOCHS,track_dict["bgrad"], track_dict["zgrad"], track_dict["vgrad"], track_dict["agrad"])
    
    fpath = r"state_dicts/training_experiment"
    save_fname = f"batch=141-epoch=15-gradclip=1-riemann=5-std=0.025-pretrain-bzv-add-a-init={seed}"
    ext = ".pth"

    torch.save(net.state_dict(), os.path.join(fpath, save_fname + ext))

    compare_path = os.path.join(fpath, save_fname + ext)
    compare_rates.compare_intensity_rates_acc(net, ns_gt, 0, 1, compare_path, training_data, test_data)
    
#%%

#plotgrad(NUM_EPOCHS, track_dict["bgrad"], track_dict["zgrad"], track_dict["vgrad"], track_dict["agrad"])

#plotres(NUM_EPOCHS, train_loss, test_loss, "NLL Loss")
#plotres(NUM_EPOCHS, track_dict["mse_train_losses"], track_dict["mse_test_losses"], "MSE Loss")
#plotgrad(NUM_EPOCHS,track_dict["bgrad"], track_dict["zgrad"], track_dict["vgrad"], track_dict["agrad"])
#plotgrad(NUM_EPOCHS, track_dict["zgrad"])

# save_fname = "EXP2-unfreeze-z-to-unfreeze-all-gradclip=30"

# torch.save(net.state_dict(), os.path.join(fpath, save_fname + ext))

# compare_path = os.path.join(fpath, save_fname + ext)

# compare_rates.compare_intensity_rates_acc(net, ns_gt, 2, 3, compare_path, training_data, test_data)

#%%
#batch_size=157
#init_beta=6.9738
#seed=7

lfpath = r"state_dicts/training_experiment"
lfname = r"batch=141-epoch=15-gradclip=1-riemann=5-std=0.025-pretrain-bzv-add-a-init=2"
ext = ".pth"

#fpath = r"state_dicts/training_experiment"
#fname = f"batch={batch_size}_LR=0.001_weight=1_acceleration_{seed}" + ".pth"
#seed=7
net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta, riemann_samples=5, node_pair_samples=6)
full_path = os.path.join(lfpath, lfname + ext)
net.load_state_dict(torch.load(full_path))
compare_rates.compare_intensity_rates_acc(net, ns_gt, 2, 3, full_path, training_data, test_data)

# est_z0 = net.a0
# est_z0 = est_z0.detach()
# plt.scatter(est_z0[:,0], est_z0[:,1], color="red", label="est")
# #plt.ylim((-0.005, 0.005))
# #plt.xlim((-0.005, 0.005))
# #plt.scatter(z_gt[:,0], z_gt[:,1], color="green", label="gt")
# plt.legend()
# plt.show()

