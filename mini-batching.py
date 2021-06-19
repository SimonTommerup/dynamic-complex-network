import torch
import nodespace
import nhpp_mod
import time
import numpy as np
import torch.nn as nn
import scipy.special as sps

np.random.seed(0)

# Define pi
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Create dynamical system with constant velocity
ns_gt = nodespace.NodeSpace()
ns_gt.beta = 7.5
ns_gt.alpha = 0.1

z_gt = np.array([[-6., 0.], [6.,1.], [0.,6.], [0.,-6.]])
v_gt = np.array([[.9,0.1], [-1.,-0.1], [0.1,-0.9], [-0.1, 0.9]])
n_points=len(z_gt)
#%%

#%%

a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
ns_gt.init_conditions(z_gt, v_gt, a_gt)

# Simulate event time data set for the two nodes
t = np.linspace(0, 15)
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


num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

print("Len training:", len(training_data))
print("Len training:", len(test_data))


def eint(i, j, z, v, t0, tn, alpha, beta): # np version
    a = z[i,0] - z[j,0]
    b = z[i,1] - z[j,1]
    m = v[i,0] - v[j,0]
    n = v[i,1] - v[j,1]
    return -np.sqrt(np.pi)*np.exp((-(a*n - b*m)**2*alpha + (m**2 + n**2)*beta)/(m**2 + n**2))*(sps.erf(alpha*((m**2 + n**2)*t0 + a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))) - sps.erf(alpha*((m**2 + n**2)*tn + a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))))/(2*np.sqrt(alpha*(m**2 + n**2)))

def to_long(ufloat, vfloat):
    return ufloat.long(), vfloat.long()

# %%
#TEST INTEGRAL
test_u = 1
test_v = 2
res_riem = ns_gt.lambda_int_sq_rapprox(t,test_u,test_v)
res_eint = eint(test_u, test_v, ns_gt.z0, ns_gt.v0, 0, t[-1], ns_gt.alpha, ns_gt.beta)

print(res_riem)
print(res_eint)

#%%
class SmallNet(nn.Module):
    def __init__(self, n_points, tn_train, tn_test):
        super().__init__()

        self.beta = nn.Parameter(torch.rand(size=(1,1)))
        #self.beta = nn.Parameter(torch.tensor(7.5))
        self.alpha = nn.Parameter(torch.rand(size=(1,1)))
        #self.alpha = nn.Parameter(torch.tensor(0.1))
        self.z0 = nn.Parameter(torch.rand(size=(n_points,2)))
        self.v0 = nn.Parameter(torch.rand(size=(n_points,2)))

        #self.a0 = torch.zeros(size=(n_points,2), device="cuda:0")
        self.a0 = torch.zeros(size=(n_points,2))
        self.tn_train = tn_train # last time point on time axis in simul
        self.tn_test = tn_test
        self.pdist = nn.PairwiseDistance(p=2) # euclidean

        # keep track of last event times for each node pair
        #self.prev_t = torch.zeros(size=(n_points, n_points), device="cuda:0")
        #self.prev_t = torch.zeros(size=(n_points, n_points))

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
        return self.z

    def get_sq_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d**2

    def lambda_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_sq_dist(t, u, v)
        # remove alpha.
        return torch.exp(self.beta - self.alpha*d)

    def _eval_integral(self, i, j, z, v, T, alpha, beta):
        a = z[i,0] - z[j,0]
        b = z[i,1] - z[j,1]
        m = v[i,0] - v[j,0]
        n = v[i,1] - v[j,1]

        return torch.sqrt(torch.pi)*torch.exp((-(a*n - b*m)**2*alpha + beta*(m**2 + n**2))/(m**2 + n**2))*torch.sqrt(alpha*(m**2 + n**2))*(torch.erf(alpha*((m**2 + n**2)*T + a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))) - torch.erf(alpha*(a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))))/(2*alpha*(m**2 + n**2))

    def eval_integral(self, i, j, t0, tn, z, v, alpha, beta):
        a = z[i,0] - z[j,0]
        b = z[i,1] - z[j,1]
        m = v[i,0] - v[j,0]
        n = v[i,1] - v[j,1]
        return -torch.sqrt(torch.pi)*torch.exp((-(a*n - b*m)**2*alpha + (m**2 + n**2)*beta)/(m**2 + n**2))*(torch.erf(alpha*((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))) - torch.erf(alpha*((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))))/(2*torch.sqrt(alpha*(m**2 + n**2)))

    def eval_integral_sample(self, i, j, t0, tn, n_samples):
        sample_times = np.random.uniform(t0, tn, n_samples)
        int_lambda = 0.

        for t_i in sample_times:
            int_lambda += self.lambda_fun(t_i, i, j)

        interval_length = tn-t0
        int_lambda = interval_length * (1 / n_samples) * int_lambda

        return int_lambda



    def forward(self, data, t0, tn, weight=1):
        eps = 1e-7
        event_intensity = 0.
        non_event_intensity = 0.
        i=0
        for u, v, event_time in data:
            u, v = to_long(u, v) # cast to int for indexing
            event_intensity += torch.log(self.lambda_fun(event_time, u=u, v=v) + eps)

        # just redefine as beta - dist

        ind = torch.triu_indices(row=n_points, col=n_points, offset=1)
        # for u, v in zip(ind[0], ind[1]):
        #     non_event_intensity += self.eval_integral(u, v, t0, tn, self.z0, self.v0, alpha=self.alpha, beta=self.beta)

        for u, v in zip(ind[0], ind[1]):
            non_event_intensity += self.eval_integral_sample(u, v, t0, tn, n_samples=10)

        return event_intensity - weight*non_event_intensity


# %%

data_set[0]

data_set = torch.from_numpy(data_set)

num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

training_batches = np.array_split(training_data, 450)

#print(len(training_batches[-1]))

#%%

tn_train = training_data[-1][-1] # last time point in training data
tn_test = test_data[-1][-1] # last time point in test data

#%%

#device = "cuda:0"
#training_data = training_data.to(device)
#test_data = test_data.to(device)

torch.random.manual_seed(0)

net = SmallNet(n_points=n_points, tn_train=tn_train, tn_test=t[-1])
#net = net.to(device)

def nll(ll):
    return -ll

#optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(net.parameters())

train_loss=[]
test_loss=[]
num_epochs=100
for epoch in range(num_epochs):
    t0 = time.time()
    running_loss = 0.
    for idx, batch in enumerate(training_batches):
        #print(f"Batch {idx+1} of {len(training_batches)}")
        net.train()
        optimizer.zero_grad()
        output = net(batch, t0=batch[0][2], tn=batch[-1][2])
        loss = nll(output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    net.eval()
    with torch.no_grad():
        test_output = net(test_data, t0=tn_train, tn=tn_test)
        tloss = nll(test_output)

    #if epoch == 0 or (epoch+1) % 25 == 0:
    print(f"Epoch {epoch+1} train loss: {running_loss/ len(training_data)} test loss: {tloss.item() / len(test_data)}")
    print(f"time: ", time.time()-t0)
    train_loss.append(running_loss / len(training_data))
    test_loss.append(tloss.item() / len(test_data))


print("Z", net.z0)
print("V", net.v0)

#%%

import matplotlib.pyplot as plt

zest = net.z0.detach().numpy()
print("GT")
print("Z:")
print(ns_gt.z0)
print("v:")
print(ns_gt.v0)
print("beta")
print(ns_gt.beta)
print("alpha")
print(ns_gt.alpha)
print()
print("EST")
print("Z:")
print(net.z0.detach().numpy())
print("v:")
print(net.v0.detach().numpy())
print("beta")
print(net.beta.detach().numpy())
print("alpha")
print(net.alpha.detach().numpy())

plt.scatter(zest[:,0], zest[:,1], color="red", label="est")
plt.scatter(ns_gt.z0[:,0], ns_gt.z0[:,1], color="blue", label="gt")
plt.legend(loc="upper left")
plt.show()


#%%
epochs = np.arange(1,num_epochs)
trainloss = np.array(train_loss[1:])
testloss = np.array(test_loss[1:])



plt.plot(epochs,trainloss,color="blue", label="train")
plt.plot(epochs,testloss, color="red", label="test")
plt.legend()
plt.show()


# %%
model_path = "sample10-int-smaller-batches-model-100-epoch-random-z-random-alpha-beta-init.pth"
torch.save(net.state_dict(), model_path)
# %%

#%%

net.alpha
# %%

print(len(data_set))
# %%
PATH = model_path
n_points = 4
tn_train = 0.
tn_test = 15.

model = SmallNet(n_points=n_points, tn_train=tn_train, tn_test=tn_test)
model.load_state_dict(torch.load(PATH))
model.eval()

# %%


def get_times(data_set):
    times = []
    for row in data_set:
        cur_t = row[time_col]
        times.append(cur_t)
    return np.array(times)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2)

traint = get_times(training_data)
testt = get_times(test_data)

this_u = 1
this_v = 2

plot_t = [traint, testt]

gttrain = []
restrain = []
for ti in plot_t[0]:
    gttrain.append(ns_gt.lambda_sq_fun(ti, this_u, this_v))
    restrain.append(net.lambda_fun(ti, this_u, this_v))

gttest = []
restest = []
for ti in plot_t[1]:
    gttest.append(ns_gt.lambda_sq_fun(ti, this_u, this_v))
    restest.append(net.lambda_fun(ti, this_u, this_v))


ax[0].plot(plot_t[0], restrain, color="red")
ax[0].plot(plot_t[0], gttrain, color="blue")
ax[0].set_title("Train")
ax[1].plot(plot_t[1], restest, color="red")
ax[1].plot(plot_t[1],gttest, color="blue")
ax[1].set_title("Test")
plt.show()
# %%
