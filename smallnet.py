import torch
import nodespace
import nhpp_mod
import time
import numpy as np
import torch.nn as nn


torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

def nll(ll):
    return -ll

def to_long(ufloat, vfloat):
    return ufloat.long(), vfloat.long()

def single_batch_train(net, n_train, training_data, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters())
    training_losses = []
    test_losses = []
    tn_train = training_data[-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    n_test = len(test_data)

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.

        net.train()
        optimizer.zero_grad()
        output = net(training_data, t0=0, tn=tn_train)
        loss = nll(output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        net.eval()
        with torch.no_grad():
            test_output = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                

        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()
        
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses


def batch_train(net, n_train, train_batches, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters())
    training_losses = []
    test_losses = []
    tn_train = train_batches[-1][-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    n_test = len(test_data)

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.
        for idx, batch in train_batches:

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
                test_loss = nll(test_output).item()
                
        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()
        
        print(f"Epoch {epoch+1}")
        print(f"elapsed time: {current_time - start_time}" )
        print(f"train loss: {avg_train_loss}")
        print(f"test loss: {avg_test_loss}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses


def infer_beta(n_points, training_data):
    tn = training_data[-1][2]
    ind = torch.triu_indices(row=n_points, col=n_points, offset=1)
    data=training_data
    n_events = []
    for u, v in zip(ind[0], ind[1]):
        event_matches = len(data[torch.logical_and(data[:,0]==u, data[:,1]==v)])
        n_events.append(event_matches)

    n_avg = sum(n_events) / len(n_events)

    return torch.log(n_avg / tn)


class SmallNet(nn.Module):
    def __init__(self, n_points, init_beta):
        super().__init__()

        #self.beta = nn.Parameter(torch.rand(size=(1,1)))
        self.beta = init_beta
        self.alpha = torch.ones(size=(1,1))

        self.z0 = nn.Parameter(torch.rand(size=(n_points,2)))
        self.v0 = nn.Parameter(torch.rand(size=(n_points,2)))
        #self.a0 = nn.Parameter(torch.rand(size=(n_points,2)))
        self.a0 = torch.zeros(size=(n_points,2))

        
        
        self.n_points = n_points
        self.ind = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        #self.tn_train = tn_train # last time point on time axis in simul
        #self.tn_test = tn_test
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
        #eps = 1e-7
        event_intensity = 0.
        non_event_intensity = 0.
        #i=0
        for u, v, event_time in data:
            u, v = to_long(u, v) # cast to int for indexing
            #event_intensity += torch.log(self.lambda_fun(event_time, u=u, v=v) + eps)
            # redefine as simply beta - dist
            event_intensity += self.beta - self.get_sq_dist(event_time, u, v)

        for u, v in zip(self.ind[0], self.ind[1]):
            non_event_intensity += self.eval_integral(u, v, t0, tn, self.z0, self.v0, alpha=self.alpha, beta=self.beta)

        #for u, v in zip(self.ind[0], self.ind[1]):
        #    non_event_intensity += self.eval_integral_sample(u, v, t0, tn, n_samples=10)

        return event_intensity - weight*non_event_intensity
