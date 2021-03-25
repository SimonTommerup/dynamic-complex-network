import torch
import torch.utils.data
import utils
import random
import time
import numpy as np
import torch.nn as nn

class DyRep(nn.Module):
    def __init__(self, 
            dataset, 
            initial_associations, 
            n_hidden=32, 
            monte_carlo_sample_size=20,
            device="cuda:0"):
        super().__init__()
        
        self.device = device

        self.n_event_types = 2 # association, communication
        self.n_nodes = dataset.n_nodes
        self.n_hidden = n_hidden

        self.A = self.initialize_A(initial_associations)
        self.S = self.initialize_S()
        self.intensity_rates = self.initialize_intensity_rates()

        self.monte_carlo_sample_size = monte_carlo_sample_size

        self.W_S = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_R = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_t = nn.Linear(1, n_hidden, bias=False)
        self.W_h = nn.Linear(n_hidden, n_hidden) # bias true
        self.psi = nn.Parameter(0.5 * torch.ones(self.n_event_types))

        self.omega = nn.ModuleList([nn.Linear(2*n_hidden, 1) for _ in range(self.n_event_types)])
        
        self.embeddings = torch.rand(size=(self.n_nodes, self.n_hidden), device=self.device)
        self.last_event_time = torch.zeros(size=(self.n_nodes, 1), device=self.device)

    def initialize_A(self, dataframe):
        A = torch.zeros(size=(self.n_nodes, self.n_nodes), device=self.device)
        sources = [u for u in dataframe["source"]]
        destinations = [v for v in dataframe["destination"]]
        
        for u, v in zip(sources, destinations):
            A[u,v] = A[v,u] = 1.
        
        return A

    def initialize_S(self):
        A = self.A.clone()
        S = A
        for row_idx, row in enumerate(A):
            idx_neighbor = row > 0
            num_neighbor = sum(idx_neighbor)
            S[row_idx, idx_neighbor] = S[row_idx, idx_neighbor].clone() / num_neighbor
        
        return S
    
    def initialize_intensity_rates(self):
        dims = (self.n_event_types, self.n_nodes, self.n_nodes)
        
        return torch.zeros(size=dims, device=self.device)
    
    def calculate_intensity_rates(self, u, v, event_type):
        z_u = self.embeddings[u].clone()
        z_v = self.embeddings[v].clone()
        
        z_cat_uv = torch.cat((z_u, z_v), dim=1)
        z_cat_vu = torch.cat((z_v, z_u), dim=1)
        
        g_symmetric = 0.5*(self.g_fun(z_cat_uv, event_type) + self.g_fun(z_cat_vu, event_type))
        intensity_rates = self.f_fun(g_symmetric, event_type).flatten()
        
        return intensity_rates

    def g_fun(self, z_cat, event_type):
        res = []
        for idx, event_type in enumerate(event_type):
            elem = self.omega[event_type](z_cat[idx].clone())
            res.append(elem)
        res = torch.stack(res)    
        
        return res
    
    def f_fun(self, x, event_type):
        res = []
        for idx, event_type in enumerate(event_type):
            r = x[idx] / self.psi[event_type]
            r = torch.clamp(r, -75, 75) # prevent overflow
            r = self.psi[event_type] * torch.log(1 + torch.exp(r))
            
            res.append(r)
        res = torch.stack(res)

        return res
    
    def n_random_nodes_not_in_current_nodes(self, n, current_nodes):
        nodes = np.random.choice([i for i in range(self.n_nodes) if i not in current_nodes], size=n, replace=False)
        nodes = torch.tensor(nodes, dtype=torch.int64, device=self.device) # Specify dtype for tensor indexing

        return nodes

    def n_concatenate(self, n, tensor):
        return torch.cat(n*[tensor.view(-1)])

    def survival_loss(self, u, v, t, sample_size):
        L_surv = 0.0
        for u_curr, v_curr, t_curr in zip(u,v,t):
            v_others = self.n_random_nodes_not_in_current_nodes(sample_size, [u_curr, v_curr])
            u_others = self.n_random_nodes_not_in_current_nodes(sample_size, [u_curr, v_curr])
            
            u_surv = 0.0
            v_surv = 0.0
            for k in torch.tensor([0,1], device="cuda:0"):
                u_cat = utils.ncat(sample_size, u_curr)
                v_cat = utils.ncat(sample_size, v_curr)
                k_cat = utils.ncat(sample_size, k)

                u_surv += self.calculate_intensity_rates(u_cat, v_others, k_cat)
                v_surv += self.calculate_intensity_rates(v_cat, u_others, k_cat)
            # end for k
            L_surv += (u_surv + v_surv) / sample_size
        # end for curr
        return L_surv

    def update_embeddings(self, u, v, t):
        z_accumulated = []
        for iteration, (u_it, v_it, t_it) in enumerate(zip(u,v,t)):
            z_previous = self.embeddings.clone() if iteration == 0 else z_accumulated[iteration-1]
            z_updated = z_previous.clone()

            hstruct = torch.zeros(size=(2, self.n_hidden), device=self.device)
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
        self.embeddings = z_updated # update model
        return

    def update_intensity_rates(self, lambda_uvk, u, v, k):
        self.intensity_rates[k,v,u] = lambda_uvk

    def update_a_and_s(self, u, v, k):
        A_previous = self.A.clone()
        self.A[u[k < 1], v[k < 1]] = self.A[v[k < 1], u[k < 1]] = 1 # update A
        
        for u_curr, v_curr, k_curr in zip(u,v,k):
            communication_event = k_curr > 0
            association_event = k_curr < 1
            association_exists = self.A[u_curr, v_curr].clone() > 0

            if communication_event and not association_exists:
                continue

            for j in [u_curr, v_curr]: # update current nodes in turn
                i = [node for node in [u_curr, v_curr] if node != j][0] # set as other node in current event
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
                
                y_normalized = y / (torch.sum(y) + 1e-7) # normalize y
                self.S[j,:] = y_normalized # Update S
    
    def forward(self, data):
        u, v, t, k = data
        
        lambda_uvk = self.calculate_intensity_rates(u, v, k)
        L_surv = self.survival_loss(u, v, t, self.monte_carlo_sample_size)

        self.update_embeddings(u, v, t)
        self.update_intensity_rates(lambda_uvk, u, v, k)
        self.update_a_and_s(u, v, k)

        return lambda_uvk, L_surv


def negative_log_likelihood(intensity_rate, survival_term):
        component_0 = -torch.sum(torch.log(intensity_rate))
        component_1 = torch.sum(survival_term)
        return component_0 + component_1

def train(model, data, loss_fun=negative_log_likelihood, epochs=1, batch_size=200, save_state_dict=True):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    model.train() 

    epoch_rloss = torch.zeros(epochs)
    for epoch_idx, epoch in enumerate(range(epochs)):
        epoch_t0 = time.time()

        batch_rloss = torch.zeros(len(data_loader))
        for batch_idx, batch in enumerate(data_loader):
            print(f"Batch {batch_idx+1} of {len(data_loader)}")
            
            batch_t0 = time.time()
            batch_data = [elem.to(model.device) for elem in batch] # CUDA
            
            optimizer.zero_grad() 

            lambda_uvk, L_surv = model(batch)  
            loss = loss_fun(lambda_uvk, L_surv) / batch_size
            
            loss.backward() 
            optimizer.step()

            batch_rloss[batch_idx] = loss.detach() 

            model.embeddings = model.embeddings.detach()  # reset the computational graph, no backprop over previous embedding
            model.S = model.S.detach()
            
            print(f"Batch {batch_idx+1} time:", time.time()-batch_t0)
            print(f"Batch loss: {batch_rloss[batch_idx]}")
        # end for batch
        
        epoch_rloss[epoch_idx] = torch.sum(batch_rloss)
        print(f"Epoch {epoch_idx+1} time:", time.time()-epoch_t0)
        print(f"Epoch loss: {epoch_rloss[batch_idx]}")
    #end for epoch

    time_stamp = time.time()
    torch.save(model.state_dict(), "data/state_dicts/model"+f"{time_stamp}")

    return model

