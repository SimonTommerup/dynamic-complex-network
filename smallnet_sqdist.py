import torch
import nodespace
import nhpp_mod
import time
import compare_rates
import numpy as np
import torch.nn as nn

# USES SQUARE EUCLIDEAN DISTANCE

torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

def nll(ll):
    return -ll

def to_long(ufloat, vfloat):
    return ufloat.long(), vfloat.long()

def single_batch_train(net, n_train, training_data, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.025)
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
        output, train_ratio = net(training_data, t0=0, tn=tn_train)
        loss = nll(output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        net.eval()
        with torch.no_grad():
            test_output, test_ratio = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                

        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()
        
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
            #print("State dict:")
            #print(net.state_dict())
            print(f"train event to non-event ratio: {train_ratio.item()}")
            print(f"test event to non-event-ratio: {test_ratio.item()}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses

def single_batch_train_track_mse(res_gt, track_nodes, net, n_train, training_data, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.025)
    training_losses = []
    test_losses = []
    tn_train = training_data[-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    n_test = len(test_data)

    mse_train_losses = []
    mse_test_losses = []

    track_dict = {
        "mse_train_losses":[], 
        "mse_test_losses":[],
        "bgrad":[],
        "vgrad":[],
        "zgrad":[]}

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.

        net.train()
        optimizer.zero_grad()
        output, train_ratio = net(training_data, t0=0, tn=tn_train)
        loss = nll(output)

        loss.backward()

        clip_value = 30.0 
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

        bgrad = torch.mean(torch.abs(net.beta.grad.clone()))
        zgrad = torch.mean(torch.abs(net.z0.grad.clone()))
        vgrad = torch.mean(torch.abs(net.v0.grad.clone()))

        optimizer.step()

        running_loss += loss.item()

        net.eval()
        with torch.no_grad():
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
            test_output, test_ratio = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                

        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()
        
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
            print(f"mse train loss {mse_train}")
            print(f"mse test loss {mse_test}")
            #print("State dict:")
            #print(net.state_dict())
            #print(f"train event to non-event ratio: {train_ratio.item()}")
            #print(f"test event to non-event-ratio: {test_ratio.item()}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        track_dict["mse_train_losses"].append(mse_train)
        track_dict["mse_test_losses"].append(mse_test)
        track_dict["bgrad"].append(torch.mean(bgrad.detach().clone()))
        track_dict["zgrad"].append(torch.mean(zgrad.detach().clone()))
        track_dict["vgrad"].append(torch.mean(vgrad.detach().clone()))
    
    return net, training_losses, test_losses, track_dict

def single_batch_train_track_grads(net, lr, training_data):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    tn_train = training_data[-1][2] 

    batch_size = torch.tensor([len(training_data)])

    track_dict = {
        "mse_train_losses":[], 
        "mse_test_losses":[],
        "bgrad":[],
        "vgrad":[],
        "zgrad":[]}

    net.train()
    optimizer.zero_grad()
    output, train_ratio = net(training_data, t0=0, tn=tn_train)
    loss = nll(output)

    loss.backward()

    # for p in net.parameters():
    #     p.grad = p.grad / batch_size
    #clip_value = 30.0 
    #torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
    #net.z0.grad = net.z0.grad / batch_size
    #net.beta.grad = net.beta.grad / (batch_size*net.n_points)

    bgrad = torch.mean(torch.abs(net.beta.grad.clone()))
    zgrad = torch.mean(torch.abs(net.z0.grad.clone()))
    vgrad = torch.mean(torch.abs(net.v0.grad.clone()))


    track_dict["bgrad"].append(torch.mean(bgrad.detach().clone()))
    track_dict["zgrad"].append(torch.mean(zgrad.detach().clone()))
    track_dict["vgrad"].append(torch.mean(vgrad.detach().clone()))
    
    return net, track_dict

def batch_train(net, n_train, train_batches, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    training_losses = []
    test_losses = []
    tn_train = train_batches[-1][-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    n_test = len(test_data)

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.
        sum_ratio = 0.
        start_t = torch.tensor([0.0])
        for idx, batch in enumerate(train_batches):
            net.train()
            optimizer.zero_grad()
            output, ratio = net(batch, t0=start_t, tn=batch[-1][2])
            loss = nll(output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sum_ratio += ratio
            start_t = batch[-1][2]

        net.eval()
        with torch.no_grad():
            test_output, test_ratio = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                
        avg_train_loss = running_loss / n_train
        avg_train_ratio = sum_ratio / len(train_batches)
        avg_test_loss = test_loss / n_test
        current_time = time.time()

        #if epoch == 0 or (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}")
        print(f"elapsed time: {current_time - start_time}" )
        print(f"train loss: {avg_train_loss}")
        print(f"test loss: {avg_test_loss}")
        print(f"train event to non-event ratio: {avg_train_ratio.item()}")
        print(f"test event to non-event-ratio: {test_ratio.item()}")
        print("State dict:")
        print(net.state_dict())
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses


def batch_train_track_mse(res_gt, track_nodes, net, n_train, train_batches, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    training_losses = []
    test_losses = []
    mse_train_losses = []
    mse_test_losses = []

    track_dict = {
        "mse_train_losses":[], 
        "mse_test_losses":[],
        "bgrad":[],
        "vgrad":[],
        "zgrad":[]}

    tn_train = train_batches[-1][-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    n_test = len(test_data)


    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.
        sum_ratio = 0.
        start_t = torch.tensor([0.0])
        
        epoch_bgrad=[]
        epoch_zgrad=[]
        epoch_vgrad=[]

        for idx, batch in enumerate(train_batches):
            net.train()
            optimizer.zero_grad()
            output, ratio = net(batch, t0=start_t, tn=batch[-1][2])
            loss = nll(output)
            loss.backward()

            #net.beta.grad = net.beta.grad/10.0
            #net.v0.grad = net.v0.grad / 10.0
            #clip_value=30.0
            #torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

            batch_bgrad = torch.mean(torch.abs(net.beta.grad))
            batch_zgrad = torch.mean(torch.abs(net.z0.grad))
            batch_vgrad = torch.mean(torch.abs(net.v0.grad))
            epoch_bgrad.append(batch_bgrad)
            epoch_zgrad.append(batch_zgrad)
            epoch_vgrad.append(batch_vgrad)


            optimizer.step()

            running_loss += loss.item()
            sum_ratio += ratio
            start_t = batch[-1][2]

        net.eval()
        with torch.no_grad():
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
            
            test_output, test_ratio = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                
        avg_train_loss = running_loss / n_train
        avg_train_ratio = sum_ratio / len(train_batches)
        avg_test_loss = test_loss / n_test
        current_time = time.time()

        # if epoch == 0 or (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}")
        print(f"elapsed time: {current_time - start_time}" )
        print(f"train loss: {avg_train_loss}")
        print(f"test loss: {avg_test_loss}")
        print(f"mse train loss / n_train {mse_train/n_train}")
        print(f"mse test loss / n_test {mse_test/n_test}")
            #print(f"train event to non-event ratio: {avg_train_ratio.item()}")
            #print(f"test event to non-event-ratio: {test_ratio.item()}")
        print("State dict:")
        print(net.state_dict())
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        track_dict["mse_train_losses"].append(mse_train)
        track_dict["mse_test_losses"].append(mse_test)
        track_dict["bgrad"].append(torch.mean(torch.tensor(epoch_bgrad)))
        track_dict["zgrad"].append(torch.mean(torch.tensor(epoch_zgrad)))
        track_dict["vgrad"].append(torch.mean(torch.tensor(epoch_vgrad)))
    
    return net, training_losses, test_losses, track_dict

def integral_count(net, full_set, train_batches):
    split_non = 0.
    net.eval()
    with torch.no_grad():
        start_t = torch.tensor([0.0])
        for idx, batch in enumerate(train_batches):
            output, non_events = net(batch, t0=start_t, tn=batch[-1][2])
            start_t = batch[-1][2]

            split_non += non_events
    
    tn_train = training_data[-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    with torch.no_grad():
        output, non_events2 = net(training_data, t0=0, tn=tn_train)

    return split_non, non_events2

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
    def __init__(self, n_points, init_beta, non_intensity_weight):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([[init_beta]]))
        self.z0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5) 
        self.v0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5) 
        self.a0 = torch.zeros(size=(n_points,2))
        self.n_points = n_points
        self.ind = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        self.pdist = nn.PairwiseDistance(p=2) # euclidean
        self.weight = non_intensity_weight

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
        return self.z

    def get_sq_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d**2

    def get_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d

    def lambda_sq_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_sq_dist(t, u, v)
        return torch.exp(self.beta - d)

    # def lambda_fun(self, t, u, v):
    #     z = self.step(t)
    #     d = self.get_dist(t, u, v)
    #     return torch.exp(self.beta - d)

    def evaluate_integral(self, i, j, t0, tn, z, v, beta):
        a = z[i,0] - z[j,0]
        b = z[i,1] - z[j,1]
        m = v[i,0] - v[j,0]
        n = v[i,1] - v[j,1]
        return -torch.sqrt(torch.pi)*torch.exp(((-b**2 + beta)*m**2 + 2*a*b*m*n - n**2*(a**2 - beta))/(m**2 + n**2))*(torch.erf(((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(m**2 + n**2)) - torch.erf(((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(m**2 + n**2)))/(2*torch.sqrt(m**2 + n**2))

    def riemann_sum(self, i, j, t0, tn, n_samples):
        # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
        x = torch.linspace(t0.item(), tn.item(), n_samples+1)
        x_mid = (x[:-1]+x[1:])/2
        dx = (tn - t0) / n_samples
        rsum = torch.zeros(size=(1,1))

        for x_i in x_mid:
            rsum += self.lambda_sq_fun(x_i, i, j) * dx
        
        return rsum

    def monte_carlo_integral(self, i, j, t0, tn, n_samples):
        sample_times = np.random.uniform(t0, tn, n_samples)
        int_lambda = 0.

        for t_i in sample_times:
            int_lambda += self.lambda_sq_fun(t_i, i, j)

        interval_length = tn-t0
        int_lambda = interval_length * (1 / n_samples) * int_lambda

        return int_lambda

    def forward(self, data, t0, tn):
        event_intensity = 0.
        non_event_intensity = 0.
        for u, v, event_time in data:
            u, v = to_long(u, v) # cast to int for indexing
            event_intensity += self.beta - self.get_sq_dist(event_time, u, v)

        for u, v in zip(self.ind[0], self.ind[1]):
            non_event_intensity += self.evaluate_integral(u, v, t0, tn, self.z0, self.v0, beta=self.beta)
        
        # for u, v in zip(self.ind[0], self.ind[1]):
        #    non_event_intensity += self.riemann_sum(self, u, v, t0, tn, n_samples=10)

        log_likelihood = event_intensity - self.weight*non_event_intensity
        ratio = event_intensity / (self.weight*non_event_intensity)

        return log_likelihood, ratio


if __name__ == "__main__":
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


    data_set = torch.from_numpy(data_set)
    num_train_samples = int(len(data_set)*0.8)
    training_data = data_set[0:num_train_samples]
    test_data = data_set[num_train_samples:]

    n_train = len(training_data)
    print("n_train:", n_train)
    print("n_test:", len(test_data))

    init_beta = infer_beta(n_points, training_data)
    print("init_beta:", init_beta)

    gt_net = SmallNet(4, 7.5)

    gt_dict = gt_net.state_dict()
    gt_z = torch.from_numpy(z_gt)
    gt_v = torch.from_numpy(v_gt)
    gt_dict["z0"] = gt_z
    gt_dict["v0"] = gt_v

    gt_net.load_state_dict(gt_dict)

    tn_train = training_data[-1][2] # last time point in training data
    tn_test = test_data[-1][2] # last time point in test data
    
    training_batches = np.array_split(training_data, 450)
    print(type(training_batches))

    batched, non_batched = integral_count(gt_net, training_data, training_batches)
    print(batched)
    print(non_batched)


    # triem= np.linspace(0,15, 50000)
    # print("Riemann sq:", ns_gt.lambda_int_sq_rapprox(t=triem, u=2, v=3))

    # u_test, v_test = to_long(torch.tensor([2.0]), torch.tensor([3.0]))
    # t0_ = torch.tensor([0.])
    # tn_ = torch.tensor([15.])

    # print("Old Exact", gt_net.eval_integral(i=u_test, j=v_test, t0=t0_, tn=tn_, z=gt_net.z0, v=gt_net.v0, alpha=gt_net.alpha, beta=gt_net.beta))
    # print("new Exact", gt_net.new_integral(i=u_test, j=v_test, t0=t0_, tn=tn_, z=gt_net.z0, v=gt_net.v0, beta=gt_net.beta))
    # gt_net.eval()
    # with torch.no_grad():
    #     print("Train likelihood:")
    #     print(-gt_net(training_data, t0=0, tn=tn_train) / len(training_data))
    #     print("Test likelihood:")
    #     print(-gt_net(test_data, t0=tn_train, tn=tn_test) / len(test_data))

    #compare_rates.compare_intensity_rates_no_path(gt_net, ns_gt, 1, 3, training_data, test_data)


   