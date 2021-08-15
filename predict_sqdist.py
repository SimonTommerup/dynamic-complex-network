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


data_set = np.load("4-point-data-set.npy", allow_pickle=True)
data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

n_train = len(training_data)
print("n_train:", n_train)
print("n_test:", len(test_data))
n_test = len(test_data)
tn_train = training_data[-1][2] # last time point in training data
tn_test = test_data[-1][2] # last time point in test data

fpath = r"state_dicts/training_experiment"
fname = f"batch=141_LR=0.001_75epoch_init_7" + ".pth"
full_path = os.path.join(fpath, fname)

def getres(t0, tn, model, track_nodes):
    time = np.linspace(t0, tn)
    res=[]
    for ti in time:
        res.append(model.lambda_sq_fun(ti, track_nodes[0], track_nodes[1]))
    return torch.tensor(res)

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
gt_net = smallnet.SmallNet(4, 7.5, non_intensity_weight=1)
gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_net.load_state_dict(gt_dict)
gt_nt = gt_net.eval()

track_nodes = [0,3]

res_gt=[getres(0, tn_train, gt_net, track_nodes), getres(tn_train, tn_test, gt_net, track_nodes)]    

#%%
fpath = r"state_dicts/training_experiment"
ll_train_loss= []
ll_test_loss= []
mse_train_loss = []
mse_test_loss = []
for seed in [1,2,3,4,5,6,7,8,9,10]:
    print(f"Seed: {seed}")
    fname = f"batch=ntrain_LR=0.025_75epoch_init_{seed}" + ".pth"
    full_path = os.path.join(fpath, fname)
    init_beta=0.0
    n_points=4
    net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta,non_intensity_weight=1)
    net.load_state_dict(torch.load(full_path))
    net.eval()

    with torch.no_grad():
        train_loss, _ = net(training_data, 0, tn_train)
        test_loss, _ = net(test_data, tn_train, tn_test)

        res_train = []
        res_test = []
        for ti in np.linspace(0, tn_train):
            res_train.append(net.lambda_sq_fun(ti, track_nodes[0], track_nodes[1]))
        
        for ti in np.linspace(tn_train, tn_test):
            res_test.append(net.lambda_sq_fun(ti, track_nodes[0], track_nodes[1]))
    
    res_train = torch.tensor(res_train)
    res_test = torch.tensor(res_test)

    mse_train = torch.mean((res_gt[0]-res_train)**2)
    mse_test = torch.mean((res_gt[1]-res_test)**2)
    ll_train_loss.append(train_loss)
    ll_test_loss.append(test_loss)

    mse_train_loss.append(mse_train)
    mse_test_loss.append(mse_test)




#%%
res1 = np.array([a / n_test for a in ll_test_loss])
print("Mean:", np.mean(res1))
print("STD: ", np.std(res1))


#%%

init_beta=0.0
n_points=4
net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta,non_intensity_weight=1)
net.load_state_dict(torch.load(full_path))

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
gt_net = smallnet.SmallNet(4, 7.5, non_intensity_weight=1)
gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v
gt_net.load_state_dict(gt_dict)
gt_nt = gt_net.eval()


ns_gt = nodespace.NodeSpace()
ns_gt.beta = 7.5
z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
n_points=len(z_gt)
ns_gt.init_conditions(z_gt, v_gt, a_gt)

ind = np.triu_indices(n_points, k=1)

#%%

#fpath = r"state_dicts/training_experiment"
#fname = f"batch=ntrain_LR=0.001_75epoch_init_7" + ".pth"
full_path = os.path.join(fpath, fname)
init_beta=0.0
n_points=4
net = smallnet.SmallNet(n_points=n_points, init_beta=init_beta,non_intensity_weight=1)
net.load_state_dict(torch.load(full_path))


nodes = [torch.tensor([i]) for i in range(n_points)]

# Predict with 
predictions = []
actuals = []
for idx, row in enumerate(test_data):
    if (idx+1) % 500==0:
        print(f"row {idx+1} of {len(test_data)}")

    actuals.append(row[1].long())

    cur_node = row[0].long()
    cur_t = row[2]
    other_nodes = [node for node in nodes if node != cur_node]
    
    max_intensity = -np.inf
    prediction = None
    for other_node in other_nodes:
        intensity = net.lambda_sq_fun(cur_t, cur_node, other_node)
        if intensity > max_intensity:
            max_intensity = intensity
            prediction = other_node

    predictions.append(prediction)

#%%
success_count=0

for p, a in zip(predictions, actuals):
    if p == a:
        success_count += 1


print(success_count / len(predictions))

#%%

success_count=0
guesses = []
for row in test_data:
    cur_node = row[0].long()
    other_nodes = [node for node in nodes if node != cur_node]
    guess = np.random.choice(n_points - 1)

    guesses.append(other_nodes[guess])



for g, a in zip(guesses, actuals):
    if g == a:
        success_count += 1


print(success_count / len(predictions))


#%%
test_t = test_data[0][2]
for u, v in zip(ind[0], ind[1]):
    print(f"{u} and {v}")
    print(net.lambda_sq_fun(test_t, u, v))


test_p = []
actual = test_data[0][1]
cn = test_data[0][0]

other_nodes = [node for node in nodes if node != cn]

max_intensity = -np.inf
prediction = None
for other_node in other_nodes:
    print(f"{cn} and {other_node}")
    intensity = net.lambda_sq_fun(cur_t, cur_node, other_node)
    if intensity > max_intensity:
        max_intensity = intensity
        prediction = other_node
test_p.append(prediction)

print("Prediction:", test_p)
print("Actual", actual)