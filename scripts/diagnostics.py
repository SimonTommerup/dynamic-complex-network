import torch
import smallnet_sqdist as smallnet
import numpy as np

model = smallnet.SmallNet(n_points=4, init_beta=torch.tensor([1.]))
model.load_state_dict(torch.load(r"state_dicts\training_experiment\batch=141_LR=0.001_75epoch_init_7.pth"))

model_full = smallnet.SmallNet(n_points=4, init_beta=torch.tensor([1.]))
model_full.load_state_dict(torch.load(r"state_dicts\training_experiment\batch=ntrain_LR=0.025_75epoch_init_7.pth"))



model_gradclip = smallnet.SmallNet(n_points=4, init_beta=torch.tensor([1.]))
model_gradclip.load_state_dict(torch.load(r"state_dicts\training_experiment\track_batch=ntrain_LR=0.025_exact_integral_clip_grad=30_init_7.pth"))


z_b = model.state_dict()["z0"].numpy()
z_f = model_full.state_dict()["z0"].numpy()
z_gc = model_gradclip.state_dict()["z0"].numpy()
z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])

import matplotlib.pyplot as plt

plt.scatter(z_b[:,0], z_b[:,1], color="red", label="batch")
plt.scatter(z_f[:,0], z_f[:,1], color="blue", label="full")
plt.scatter(z_gc[:,0], z_gc[:, 1], color="black", label="full gradclip")
plt.scatter(z_gt[:,0], z_gt[:, 1], color="green", label="gt")
plt.legend()
plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

data_set = np.load("4-point-data-set.npy", allow_pickle=True)
data_set = torch.from_numpy(data_set)
num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]
training_batches = np.array_split(training_data, 450)

tn_train = training_data[-1][2]
#log_lik, ratio = model_full(training_data, t0=0, tn=tn_train)


# first batch
first_batch = training_batches[0]
tn_batch=first_batch[-1][2]

# %%

net = model
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
torch.manual_seed(0)

net.train()
optimizer.zero_grad()
#output, train_ratio = net(training_data, t0=0, tn=tn_train)
output, train_ratio = net(first_batch, t0=0, tn=tn_batch)
loss = -output
loss.backward()


total_norm=0.0
for p in net.parameters():
    param_norm = p.grad.data.norm(2)
    print(param_norm)
    total_norm += param_norm.item() ** 2

total_norm = total_norm ** (1. / 2)
print(total_norm)


print("Gradients:")
print("Beta")
print(net.beta.grad)
print("Z")
print(net.z0.grad)
print("V")
print(net.v0.grad)
print("ssq grad")
print(torch.sum(torch.sqrt(net.z0.grad**2)))

#optimizer.step()
# %%
torch.mean(net.v0.grad)
# %%

z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])

gt_net = smallnet.SmallNet(4, 7.5)

gt_dict = gt_net.state_dict()
gt_z = torch.from_numpy(z_gt)
gt_v = torch.from_numpy(v_gt)
gt_dict["z0"] = gt_z
gt_dict["v0"] = gt_v

gt_net.load_state_dict(gt_dict)

#%%
gt_net.eval()
model.eval()
model_full.eval()
model_gradclip.eval()
t = np.linspace(0, tn_train)
res_gt = []
res_md = []
res_fu = []
res_gc = []
for ti in t:
    res_gt.append(gt_net.lambda_sq_fun(ti, 2, 3))
    res_md.append(model.lambda_sq_fun(ti, 2, 3))
    res_fu.append(model_full.lambda_sq_fun(ti, 2, 3))
    res_gc.append(model_gradclip.lambda_sq_fun(ti, 2, 3))

def to_arr(reslist):
    return torch.tensor(reslist)



res_gt = to_arr(res_gt)
res_md = to_arr(res_md)
res_fu = to_arr(res_fu)

plt.plot(t, res_gt)
plt.plot(t, res_md)
plt.plot(t, res_fu)
plt.show()

#%%

mse0 = torch.mean((res_gt-res_md)**2)
mse1 = torch.mean((res_gt-res_fu)**2)
print("MSE gt and batch:", mse0)
print("MSE gt and full:", mse1)
#%%

a = gt_net.lambda_sq_fun(0, 2, 3)
print(a.shape)
torch.reshape(a, shape=(1,1))
print(a.shape)
# %%
