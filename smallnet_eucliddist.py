import torch
import nodespace
import nhpp
import time
import compare_rates
import numpy as np
import torch.nn as nn


torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

def nll(ll):
    return -ll

def to_long(ufloat, vfloat):
    return ufloat.long(), vfloat.long()

def single_batch_train(net, n_train, training_data, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.025)
    training_losses = []
    test_losses = []
    tn_train = training_data[-1][-1] # last time point in training data
    tn_test = test_data[-1][-1] # last time point in test data
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
            print("State dict:")
            print(net.state_dict())
            #print(f"train event to non-event ratio: {train_ratio.item()}")
            #print(f"test event to non-event-ratio: {test_ratio.item()}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses

def batch_train(net, n_train, training_data, train_loader, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    training_losses = []
    test_losses = []
    tn_train = training_data[-1][-1] # last time point in training data
    tn_test = test_data[-1][-1] # last time point in test data
    n_test = len(test_data)

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.
        start_t = torch.tensor([0.0])
        for idx, batch in enumerate(train_loader):
            if (idx+1)==1 or (idx+1) % 100 == 0:
                print(f"Batch {idx+1} of {len(train_loader)}")
                
            net.train()
            optimizer.zero_grad()

            output = net(batch, t0=start_t, tn=batch[-1][2])
            loss = nll(output)
            loss.backward()

            clip_value = 30.0 
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

            optimizer.step()

            running_loss += loss.item()
            start_t = batch[-1][2]

        net.eval()
        with torch.no_grad():
            test_output = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                
        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()

        if epoch == 0 or (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses

def batch_train_scheduled(net, n_train, training_data, train_loader, test_data, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    training_losses = []
    test_losses = []
    tn_train = training_data[-1][-1] # last time point in training data
    tn_test = test_data[-1][-1] # last time point in test data
    iters = len(train_loader)
    n_test = len(test_data)

    best_loss = torch.tensor([1e100])
    save_path = r"state_dicts/training_experiment/hospital_best_state.pth"

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.
        start_t = torch.tensor([0.0])
        for idx, batch in enumerate(train_loader):
            if (idx+1)==1 or (idx+1) % 100 == 0:
                print(f"Batch {idx+1} of {len(train_loader)}")
                
            net.train()
            optimizer.zero_grad()

            output = net(batch, t0=start_t, tn=batch[-1][2])
            loss = nll(output)
            loss.backward()

            clip_value = 30.0 
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

            optimizer.step()
            scheduler.step(epoch + idx / iters)

            running_loss += loss.item()
            start_t = batch[-1][2]

        print(f"Epoch {epoch+1} Last LR: {scheduler.get_last_lr()}")

        net.eval()
        with torch.no_grad():
            test_output = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output).item()
                
        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss / n_test
        current_time = time.time()

        if avg_test_loss < best_loss:
            print("Lowest test loss.")
            #print(f"Saved model state_dict to {save_path}.")
            #best_loss = avg_test_loss
            #torch.save(net.state_dict(), save_path)

        if epoch == 0 or (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    return net, training_losses, test_losses, optimizer

def batch_train_track_mse(res_gt, track_nodes, net, n_train, train_batches, test_data, num_epochs):
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=0.001)
    training_losses = []
    test_losses = []
    mse_train_losses = []
    mse_test_losses = []

    track_dict = {
        "mse_train_losses":[], 
        "mse_test_losses":[],
        "bgrad":[],
        "vgrad":[],
        "zgrad":[],
        "agrad":[]}

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
        epoch_agrad=[]
        
        net.train()
        for idx, batch in enumerate(train_batches):
            if (idx+1) == 0 or (idx + 1) % 250 == 0:
                print(f"Batch {idx+1} of {len(train_batches)}")

            optimizer.zero_grad()
            output = net(batch, t0=start_t, tn=batch[-1][2])
            loss = nll(output)
            loss.backward()

            clip_value = 30.0 
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

            batch_bgrad = torch.mean(torch.abs(net.beta.grad))
            batch_zgrad = torch.mean(torch.abs(net.z0.grad))
            batch_vgrad = torch.mean(torch.abs(net.v0.grad))
            batch_agrad = torch.mean(torch.abs(net.a0.grad))
            epoch_bgrad.append(batch_bgrad)
            epoch_zgrad.append(batch_zgrad)
            epoch_vgrad.append(batch_vgrad)
            epoch_agrad.append(batch_agrad)



            optimizer.step()

            running_loss += loss.item()
            start_t = batch[-1][2]

        net.eval()
        with torch.no_grad():
            res_train = []
            res_test = []
            for ti in np.linspace(0, tn_train):
                res_train.append(net.lambda_fun(ti, track_nodes[0], track_nodes[1]))
            
            for ti in np.linspace(tn_train, tn_test):
                res_test.append(net.lambda_fun(ti, track_nodes[0], track_nodes[1]))
            
            res_train = torch.tensor(res_train)
            res_test = torch.tensor(res_test)

            mse_train = torch.mean((res_gt[0]-res_train)**2)
            mse_test = torch.mean((res_gt[1]-res_test)**2)
            
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
            print(f"mse train loss {mse_train}")
            print(f"mse test loss {mse_test}")
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
        track_dict["agrad"].append(torch.mean(torch.tensor(epoch_agrad)))
    
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
        inner = torch.logical_and(data[:,0]==u, data[:,1]==v)
        var = data[torch.logical_and(data[:,0]==u, data[:,1]==v)]
        event_matches = len(data[torch.logical_and(data[:,0]==u, data[:,1]==v)])
        n_events.append(event_matches)

    n_avg = sum(n_events) / len(n_events)

    return torch.log(n_avg / tn)


class SmallNet(nn.Module):
    def __init__(self, pars_mode, n_points, init_beta, riemann_samples, node_pair_samples, non_intensity_weight=1):
        super().__init__()

        self.beta = nn.Parameter(torch.tensor([[init_beta]]), requires_grad=True)
        self.z0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=False)
        self.v0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=False)
        self.a0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=False)

        if pars_mode == 1:
            self.z0 = nn.Parameter(self.init_parameter(self.z0), requires_grad=True)
        elif pars_mode == 2:
            self.z0 = nn.Parameter(self.init_parameter(self.z0), requires_grad=True)
            self.v0 = nn.Parameter(self.init_parameter(self.v0), requires_grad=True)
        elif pars_mode == 3:
            self.z0 = nn.Parameter(self.init_parameter(self.z0), requires_grad=True)
            self.v0 = nn.Parameter(self.init_parameter(self.v0), requires_grad=True)
            self.a0 = nn.Parameter(self.init_parameter(self.a0), requires_grad=True)

        self.n_points = n_points
        self.n_node_pairs = n_points*(n_points-1) // 2

        self.ind = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        self.pdist = nn.PairwiseDistance(p=2) # euclidean
        self.integral_samples=riemann_samples
        self.node_pair_samples = node_pair_samples
        self.non_event_weight = non_intensity_weight

    def init_beta(self, tensor):
        return torch.nn.init.uniform_(tensor, a=0, b=0.025)

    def init_parameter(self, tensor):
        #a,b = shape, r1,r2 = range
        #torch.FloatTensor(1, 1).uniform_(-0.025, 0.025)
        #return torch.nn.init.normal_(tensor, mean=0.0, std=0.025)
        return torch.nn.init.uniform_(tensor, a=-0.025, b=0.025)

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
        return self.z

    def get_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d

    def lambda_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_dist(t, u, v)
        return torch.exp(self.beta - d)

    def riemann_sum(self, i, j, t0, tn, n_samples):
        # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
        x = torch.linspace(t0.item(), tn.item(), n_samples+1)
        x_mid = (x[:-1]+x[1:])/2
        dx = (tn - t0) / n_samples
        rsum = torch.zeros(size=(1,1))

        for x_i in x_mid:
            rsum += self.lambda_fun(x_i, i, j) * dx
        
        return rsum

    def monte_carlo_integral(self, i, j, t0, tn, n_samples):
        sample_times = np.random.uniform(t0, tn, n_samples)
        int_lambda = 0.

        for t_i in sample_times:
            int_lambda += self.lambda_fun(t_i, i, j)

        interval_length = tn-t0
        int_lambda = interval_length * (1 / n_samples) * int_lambda

        return int_lambda

    def get_unique_links(self, batched_data):
        unique_links = torch.unique(batched_data[:,:-1], dim=0)
        return unique_links

    def forward(self, data, t0, tn):
        event_intensity = 0.
        non_event_intensity = 0.

        for u, v, event_time in data:
            u, v = to_long(u, v) # cast to int for indexing
            event_intensity += self.beta - self.get_dist(event_time, u, v)

        triu_samples = torch.randperm(self.n_node_pairs)[:self.node_pair_samples]
        for idx in triu_samples:
            u, v = self.ind[0][idx], self.ind[1][idx]
            non_event_intensity += self.riemann_sum(u, v, t0, tn, n_samples=self.integral_samples)

        log_likelihood = event_intensity - self.non_event_weight*non_event_intensity

        return log_likelihood


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