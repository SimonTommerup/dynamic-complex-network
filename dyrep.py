import torch
import collections
import torch.utils.data
import torch.nn as nn
import random
import numpy as np

class DyRep(nn.Module):
    def __init__(self, 
            dataset, 
            initial_associations, 
            n_hidden, 
            bipartite=False, 
            monte_carlo_sample_size=20,
            device="cuda:0"):
        super().__init__()
        
        self.device = device
        self.n_event_types = 2
        self.n_nodes = dataset.n_nodes

        self.A = self.init_mat_a(initial_associations)
        self.S = self.init_mat_s(self.A.clone())
        self.intensity_rates = self.init_mat_l()

        self.monte_carlo_sample_size = monte_carlo_sample_size
        self.n_hidden = n_hidden

        self.W_S = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_R = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_t = nn.Linear(1, n_hidden, bias=False)
        self.W_h = nn.Linear(n_hidden, n_hidden) # bias true
        self.psi = nn.Parameter(0.5 * torch.ones(self.n_event_types))

        self.omega = nn.ModuleList([nn.Linear(2*n_hidden, 1) for _ in range(self.n_event_types)])
        self.last_event_time = torch.zeros(size=(self.n_nodes, 1), device=self.device)
        
        self.embeddings = torch.rand(size=(self.n_nodes, self.n_hidden), device=self.device)
        
    def init_mat_a(self, data):
        mat_a = torch.zeros(size=(self.n_nodes, self.n_nodes), device=self.device)
        sources = [u for u in data["source"]]
        destinations = [v for v in data["destination"]]
        for u, v in zip(sources, destinations):
            mat_a[u,v] = mat_a[v,u] = 1.
        return mat_a
    
    def init_mat_s(self, mat_a):
        temp_s = mat_a.clone()
        for row_idx, row in enumerate(mat_a):
            neighbor_idx = row > 0
            neighbor_num = sum(neighbor_idx)
            temp_s[row_idx, neighbor_idx] = temp_s[row_idx, neighbor_idx].clone() / neighbor_num
        mat_s = temp_s.clone()
        return mat_s
    
    def init_mat_l(self):
        dims = (self.n_event_types, self.n_nodes, self.n_nodes)
        return torch.zeros(size=dims, device=self.device)
    
    def calculate_intensity_rates(self, u, v, event_type):
        z_u = self.embeddings[u].clone()
        z_v = self.embeddings[v].clone()
        z_cat_uv = torch.cat((z_u, z_v), dim=1)
        z_cat_vu = torch.cat((z_v, z_u), dim=1)
        g_symmetric = 0.5*(self.g_k(z_cat_uv, event_type) + self.g_k(z_cat_vu, event_type))
        f_res = self.f_k(g_symmetric, event_type)

        return f_res.flatten()
    
    def g_k(self, z_cat, event_type):
        res = []
        for idx, event_type in enumerate(event_type):
            elem = self.omega[event_type](z_cat[idx].clone())
            res.append(elem)
        return torch.stack(res)

    def f_k(self, g_k_res, event_type):
        res = []
        for idx,event_type in enumerate(event_type):
            inner_inner_expr = g_k_res[idx].clone() / self.psi[event_type].clone()
            inner_inner_expr = torch.clamp(inner_inner_expr.clone(), -75, 75)
            inner_expr = 1 + torch.exp(inner_inner_expr.clone())
            elem = self.psi[event_type].clone() * torch.log(inner_expr.clone())
            res.append(elem)
        return torch.stack(res)

    def survival_loss(self, u, v, t, sample_size):
        L_surv = 0.0

        for u_curr, v_curr, t_curr in zip(u,v,t):
            v_others = torch.tensor(np.random.choice([i for i in range(self.n_nodes) if i not in [u_curr, v_curr]], size=sample_size, replace=False), dtype=torch.int64, device=self.device)
            u_others = torch.tensor(np.random.choice([i for i in range(self.n_nodes) if i not in [u_curr, v_curr]], size=sample_size, replace=False), dtype=torch.int64, device=self.device)
            
            u_surv = 0.0
            v_surv = 0.0
            for k in torch.tensor([0,1], device="cuda:0"):
                u_surv = u_surv + self.calculate_intensity_rates(torch.cat(sample_size*[u_curr.view(-1)]), v_others, torch.cat(sample_size*[k.view(-1)]))
                v_surv = u_surv + self.calculate_intensity_rates(torch.cat(sample_size*[v_curr.view(-1)]), u_others, torch.cat(sample_size*[k.view(-1)]))

            L_surv = L_surv + (u_surv + v_surv) / sample_size
        
        return L_surv

    def update_embeddings_first(self, u, v, t):
        # L: Localized Embedding Propagation 
        # S: Self-propagation 
        # E: Exogenous drive 
        # Inplace error must connect to here, since IF L+S+E = 0, then no problem.
        print("Update embeddings")

        for u_curr, v_curr, t_curr in zip(u,v,t):
            current_embeddings = self.embeddings.detach().clone()
            current_last_event_time = self.last_event_time.detach().clone()
            #print("Iter")
            for j in [u_curr, v_curr]:
                not_j = [node.clone() for node in [u_curr, v_curr] if node != j][0]
                
                L = self.W_S(self.localized_embedding_propagation(not_j, current_embeddings))
                #S = self.W_R(current_embeddings[j].clone())
                #print("HER")
                #TEST_TIME = t_curr.clone() - self.last_event_time[j].clone()
                #print("TIME SHAPE: ", TEST_TIME.shape)
                #E = self.W_t(TEST_TIME)
                E = torch.tensor([0.], device=self.device)

                z_j = torch.sigmoid(L + E.clone())
                #z_j = torch.sigmoid(L + S + E)
                #embeddings = self.embeddings.clone()
                current_embeddings[j] = z_j # inplace modification ? 
                #self.embeddings = embeddings.clone()

                #last_event_time = self.last_event_time.clone()
                #last_event_time[j] = t_curr # inplace modification ?
                current_last_event_time[j] = t_curr
                #self.last_event_time = last_event_time.clone()
                
            self.embeddings = current_embeddings.clone()
            self.last_event_time = current_last_event_time.clone()

    def update_embeddings(self, u, v, t):
        z_accumulated = []
        for iteration, (u_it, v_it, t_it) in enumerate(zip(u,v,t)):
            z_previous = self.embeddings.clone() if iteration == 0 else z_accumulated[iteration-1]
            z_updated = z_previous.clone()

            hstruct = z_previous.new(2, self.n_hidden).fill_(0)
            time_deltas = torch.zeros(2, device=self.device)

            for k, (n1, n2, t_curr) in enumerate(zip([u_it, v_it], [v_it, u_it], [t_it, t_it])):
                # Equation (4) for all nodes in batch
                n1_tbar = self.last_event_time[n1].clone()
                time_deltas[k] = t_curr - n1_tbar
                n2_neighbors = self.A[n2, :].clone() > 0
                n2_n_neighbors = torch.sum(n2_neighbors)
            
                if n2_n_neighbors > 0:
                    h_n2 = self.W_h(z_previous[n2_neighbors]).view(n2_n_neighbors, self.n_hidden) # views and all
                    q_n2_i = torch.exp(self.S[n2, n2_neighbors].clone()).view(n2_n_neighbors, 1)
                    q_n2_i = q_n2_i / (torch.sum(q_n2_i) + 1e-7)
                    hstruct[k, :] = torch.max(torch.sigmoid(q_n2_i * h_n2),dim=0)[0].view(1, self.n_hidden)
                
                self.last_event_time[n1] = t_curr
                # end for k

            L = self.W_S(hstruct.view(2, self.n_hidden))
            S = self.W_R(z_previous[[n1, n2],:].view(2,-1))
            E = self.W_t(time_deltas.view(2,1))

            z_updated[[n1, n2],:] = torch.sigmoid(L + S + E)
            z_accumulated.append(z_updated)
                
        # end for iteration
        self.embeddings = z_updated # for updating model state with last update
        return

    def localized_embedding_propagation(self, u, current_embeddings):
        neighbors = self.A[u, :].clone() > 0
        neighbor_embeddings = current_embeddings[neighbors].clone()
        h_struct_u = torch.zeros(self.n_hidden, device=self.device)

        if torch.sum(neighbors) > 0:
            test = neighbor_embeddings.clone()
            #print(test.shape)
            h_i = self.W_h(neighbor_embeddings.clone())
            q_ui = self.attention(u.clone(), neighbors.clone())
            h_struct_u = torch.max(torch.sigmoid(q_ui * h_i), dim=0).values
        
        return h_struct_u
   
    def attention(self, u, neighbors):
        n_neighbors = sum(neighbors)
        smoothing = 1e-10
        numerators = torch.exp(self.S[u, neighbors].clone())
        denominators = torch.sum(numerators.clone()) + smoothing
        q_ui = numerators.clone() / denominators.clone()
        return q_ui.view(n_neighbors, 1)

    def update_intensity_rates(self, lambda_uvk, u, v, k):
        #print("Update intensity rates")
        #intensity_rates = self.intensity_rates.clone()
        self.intensity_rates[k,v,u] = lambda_uvk
        #self.intensity_rates = intensity_rates.clone()

    def update_a_and_s(self, u, v, k):
        # Update A for batch
        A_previous = self.A.clone()
        A_updated = A_previous
        A_updated[u[k < 1], v[k < 1]] = A_updated[v[k < 1], u[k < 1]] = 1
        self.A = A_updated.clone()
        
        # Update S for node pairs
        for u_curr, v_curr, k_curr in zip(u,v,k):
            communication_event = k_curr > 0
            association_event = k_curr < 1
            association_exists = self.A[u_curr, v_curr].clone() > 0

            if communication_event and not association_exists:
                continue

            for j in [u_curr, v_curr]:
                i = [node for node in [u_curr, v_curr] if node != j][0]

                b = 1 / torch.sum(self.A[j,:].clone() > 0)
                y = self.S[j, :].clone()

                if communication_event and association_exists:
                    y[i] = b + self.intensity_rates[k_curr, j, i].clone()
                
                elif association_event:
                    b_prime = 1 / torch.sum(A_previous[j,:]>0)
                    x = b_prime - b
                    
                    w = (y != 0)
                    w[i] = False

                    y[i] = b + self.intensity_rates[k_curr, j, i].clone()
                    y[w] = y[w].clone() - x
                
                y = y.clone() / (torch.sum(y.clone()) + 1e-7) # normalize y
                S = self.S.clone()
                S[j,:] = y # inplace modification ?
                self.S = S.clone()

    def forward(self, data):
        u, v, t, k = data
        
        lambda_uvk = self.calculate_intensity_rates(u, v, k)
        L_surv = self.survival_loss(u, v, t, self.monte_carlo_sample_size)
        
        self.update_embeddings(u, v, t)
        self.update_intensity_rates(lambda_uvk, u, v, k)
        self.update_a_and_s(u, v, k)

        return lambda_uvk, L_surv

def negative_log_likelihood(lambda_uvk, L_surv):
    L1 = -torch.sum(torch.log(lambda_uvk)) / 200.
    L2 = torch.sum(L_surv) / 200.
    return L1 + L2

if __name__=="__main__":
    import pandas as pd
    import time
    from torch.utils.data import DataLoader
    from data_sets import MITDataSet
    
    device = "cuda:0"

    initial_associations = pd.read_csv(r"data\soc-evo\pre-processed\soc-evo-initial-associations.csv")
    training_data = MITDataSet(r"data\soc-evo\pre-processed\soc-evo-train-data-set.csv")

    training_data_loader = DataLoader(training_data, batch_size=200)

    model = DyRep(training_data, initial_associations, n_hidden=32, bipartite=False, monte_carlo_sample_size=20, device=device)
    model.to(model.device)
    model.train()
    # smaller learning rate, learning rate scheduler, SGD optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    batch_loss = torch.zeros(len(training_data_loader))
    
    e_t0 = time.time()
    
    for idx, batch in enumerate(training_data_loader):
        print(f"Batch {idx+1} of {len(training_data_loader)}")
        
        batch = [elem.to(device) for elem in batch] # CUDA
        
        b_t0 = time.time()
        optimizer.zero_grad()
        
        lambda_uvk, L_surv = model(batch)
        loss = negative_log_likelihood(lambda_uvk, L_surv)
        
        loss.backward()
        optimizer.step()

        batch_loss[idx] = loss.detach() # should be detached. cpu mem leak ?
        # detach A ?
        model.embeddings = model.embeddings.detach()  # to reset the computational graph and avoid backpropagating second time
        model.S = model.S.detach()
        
        print(f"Batch {idx+1} time:", time.time()-b_t0)
        print(f"Batch loss: {batch_loss[idx]}")
        
    e_t1 = time.time()
    e_time = e_t1 - e_t0
    print("Epoch time:", e_time)
