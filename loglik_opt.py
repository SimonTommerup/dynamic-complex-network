import torch
import nodespace
import nhpp_mod
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

z_gt = np.array([[-6, 0], [6.,1.]])
v_gt = np.array([[1,0.], [-1,0.]])


a_gt = np.array([[0.,0.], [0.,0.]])
ns_gt.init_conditions(z_gt, v_gt, a_gt)

# Simulate event time data set for the two nodes
t = np.linspace(0, 15)
rmat = nhpp_mod.root_matrix(ns_gt) 
mmat = nhpp_mod.monotonicity_mat(ns_gt, rmat)


nhppmat = nhpp_mod.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

# create a data list like (u, v, event time) sort by event time


data_set = nhpp_mod.get_entry(nhppmat, u=0, v=1) # event times

#%%
len(data_set)

#%%


# Intensity = lambda = exp ( beta - alpha * sqdist(z_u, z_v))

num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]


def llint(z, v, T, alpha, beta):
    a = z[0,0] - z[1,0]
    b = z[0,1] - z[1,1]
    m = v[0,0] - v[1,0]
    n = v[0,1] - v[1,1]
    return np.sqrt(np.pi)*np.exp((-(a*n - b*m)**2*alpha + beta*(m**2 + n**2))/(m**2 + n**2))*np.sqrt(alpha*(m**2 + n**2))*(sps.erf(alpha*((m**2 + n**2)*T + a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))) - sps.erf(alpha*(a*m + b*n)/np.sqrt(alpha*(m**2 + n**2))))/(2*alpha*(m**2 + n**2))

def loglikelihood(ns, event_time, T, weight=1):
    event_intensity = 0.
    for t in event_time:
        event_intensity += ns.lambda_sq_fun(t, 0, 1)
    
    non_event_intensity = llint(ns.z0, ns.v0, T, ns.alpha, ns.beta)
    
    return np.log(event_intensity) - weight*non_event_intensity

loglik = loglikelihood(ns_gt, training_data, t[-1], weight=1e3)


#%%
class SmallNet(nn.Module):
    def __init__(self, last_time_point):
        super().__init__()
        self.beta = nn.Parameter(torch.rand(size=(1,1)))
        self.alpha = nn.Parameter(torch.rand(size=(1,1)))
        self.z0 = nn.Parameter(torch.rand(size=(2,2)))
        #self.z0 = nn.Parameter(torch.from_numpy(np.array([[-4, 0.1], [4.,0.9]])))
        self.v0 = nn.Parameter(torch.rand(size=(2,2)))
        #self.v0 =  nn.Parameter(torch.from_numpy(np.array([[0.8,0.], [-0.8,0.]])))
        self.a0 = torch.zeros(size=(2,2))
        self.T = last_time_point
        self.pdist = nn.PairwiseDistance(p=2) # euclidean
    
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
        return torch.exp(self.beta - self.alpha*d)

    def eval_integral(self, z, v, T, alpha, beta):
        a = z[0,0] - z[1,0]
        b = z[0,1] - z[1,1]
        m = v[0,0] - v[1,0]
        n = v[0,1] - v[1,1]

        erf=torch.erf(alpha*(a*m + b*n))
        res= torch.sqrt(torch.pi)*torch.exp((-(a*n - b*m)**2*alpha + beta*(m**2 + n**2))/(m**2 + n**2))*torch.sqrt(alpha*(m**2 + n**2))*(torch.erf(alpha*((m**2 + n**2)*T + a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))) - torch.erf(alpha*(a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))))/(2*alpha*(m**2 + n**2))
        if torch.isnan(res):
            debug=True
        return torch.sqrt(torch.pi)*torch.exp((-(a*n - b*m)**2*alpha + beta*(m**2 + n**2))/(m**2 + n**2))*torch.sqrt(alpha*(m**2 + n**2))*(torch.erf(alpha*((m**2 + n**2)*T + a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))) - torch.erf(alpha*(a*m + b*n)/torch.sqrt(alpha*(m**2 + n**2))))/(2*alpha*(m**2 + n**2))

    
    def forward(self, data, weight=1e3):
        eps = 1e-7
        event_intensity = 0.
        for event_time in data:
            event_intensity += torch.log(self.lambda_fun(event_time, u=0, v=1) + eps)

        non_event_intensity = self.eval_integral(self.z0, self.v0, self.T, self.alpha, self.beta)

        if torch.isnan(event_intensity) or torch.isnan(non_event_intensity):
            debug=True

        return event_intensity - weight*non_event_intensity


# %%
data_set = torch.from_numpy(data_set)

num_train_samples = int(len(data_set)*0.8)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]

# (event-time, u, v)
# sum up 

net = SmallNet(last_time_point=t[-1])
#net.load_state_dict(torch.load("trained_model.pth"))

def nll(ll):
    res = -ll
    if torch.isnan(res):
        debug=True
    return -ll

#optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(net.parameters())

train_loss=[]
test_loss=[]
num_epochs=500
for epoch in range(num_epochs):

    net.train()
    optimizer.zero_grad()
    output = net(training_data)
    loss = nll(output)
    loss.backward()
    optimizer.step()

    net.eval()
    with torch.no_grad():
        test_output = net(test_data)
        tloss = nll(test_output)

    if epoch == 0 or (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1} train loss: {loss.item()} test loss: {tloss.item()}")
    train_loss.append(loss.item())
    test_loss.append(tloss.item())


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
torch.save(net.state_dict(), "trained_model-all-pars.pth")
# %%

#%%

net.alpha
# %%
