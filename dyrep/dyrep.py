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
            n_event_types=2,
            monte_carlo_sample_size=10,
            device="cuda:0"):
        super().__init__()
        
        self.device = device

        self.n_event_types = n_event_types # association, communication
        self.n_nodes = dataset.n_nodes
        self.n_hidden = n_hidden

        A = self.initialize_A(initial_associations)
        self.register_buffer("A", A)

        S = self.initialize_S()
        self.register_buffer("S", S)

        intensity_rates = self.initialize_intensity_rates()
        self.register_buffer("intensity_rates", intensity_rates)

        self.monte_carlo_sample_size = monte_carlo_sample_size

        self.W_S = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_R = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_t = nn.Linear(1, n_hidden, bias=False)
        self.W_h = nn.Linear(n_hidden, n_hidden) # bias true
        self.psi = nn.Parameter(0.5 * torch.ones(self.n_event_types))

        self.omega = nn.ModuleList([nn.Linear(2*n_hidden, 1) for _ in range(self.n_event_types)])
        
        embeddings = torch.rand(size=(self.n_nodes, self.n_hidden), device=self.device)
        self.register_buffer("embeddings", embeddings)

        last_event_time = torch.zeros(size=(self.n_nodes, 1), device=self.device)
        self.register_buffer("last_event_time", last_event_time)

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

    def dyrep_survival(self, u, v, t, sample_size):
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

    def poisson_survival(self, u, v, t, sample_size):
        len_batch = u.shape[0]
        surv_batch = torch.zeros(size=(len_batch,1))
        node_list = torch.unique(torch.cat((u,v)))
        idx = 0
        for u_cur, v_cur, t_cur in zip(u,v,t):
            surv_cur = 0.0
            u_surv = 0.0
            v_surv = 0.0
            i = 0

            while i < sample_size:
                u_other = self.n_random_nodes_not_in_current_nodes(1, [u_cur, v_cur])
                v_other = self.n_random_nodes_not_in_current_nodes(1, [u_cur, v_cur])
                #u_other = random.choice(node_list)
                #v_other = random.choice(node_list)

                if u_other in [u_cur, v_cur] or v_other in [u_cur, v_cur]:
                    continue
                
                i += 1 
                
                for k in torch.tensor([0,1], device="cuda:0"):
                    # multiply by last time (t2-t1)?
                    u_surv += self.calculate_intensity_rates(u_cur.view(-1), v_other.view(-1), k.view(-1))
                    v_surv += self.calculate_intensity_rates(v_cur.view(-1), u_other.view(-1), k.view(-1)) 

            surv_cur += (u_surv + v_surv) / sample_size
            surv_batch[idx] = surv_cur
            idx += 1

        return surv_batch
        
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
                    save_0 = self.S[n2, n2_neighbors].clone()
                    q_n2_i = torch.exp(self.S[n2, n2_neighbors].clone()).view(n2_n_neighbors, 1)
                    save_1 = q_n2_i.clone()
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
        for u_curr, v_curr, k_curr in zip(u,v,k):
            A_previous = self.A.clone()
            if k_curr < 1:
                self.A[u_curr, v_curr] = self.A[v_curr, u_curr] = 1 # update A 

            communication_event = k_curr > 0
            association_event = k_curr < 1
            association_exists = self.A[u_curr, v_curr].clone() > 0

            if communication_event and not association_exists:
                continue

            for j in [u_curr, v_curr]: # update current nodes in turn
                i = [node for node in [u_curr, v_curr] if node != j][0] # set as other node in current event
                
                n_new_neighbors = torch.sum(self.A[j,:].clone() > 0)
                if n_new_neighbors == 0:
                    b = 0
                else:
                    b = 1 / (n_new_neighbors + 1e-7)

                y = self.S[j, :].clone()

                if communication_event and association_exists:
                    y[i] = b + self.intensity_rates[k_curr, j, i].clone()
                
                elif association_event:
                    n_prev_neighbors = torch.sum(A_previous[j, :] > 0)

                    if n_prev_neighbors == 0:
                        b_prime = 0
                    else:
                        b_prime = 1 / (n_prev_neighbors + 1e-7)
                    
                    x = b_prime - b # greater than or equal to zero
                    
                    w = (y != 0)
                    w[i] = False
                    y[i] = b + self.intensity_rates[k_curr, j, i].clone()
                    y[w] = y[w].clone() - x
                
                
                y_normalized = y / (torch.sum(y) + 1e-7) # normalize y
                
                # the values can get negative if x > y[w], 
                # but DyReP paper says that any value in S is [0,1]
                #y_normalized = torch.clamp(y_normalized, min=0, max=1)
                self.S[j,:] = y_normalized # Update S
    
    def forward(self, data):
        u, v, t, k = data
        
        output_intensity = self.calculate_intensity_rates(u, v, k)
        output_survival = self.poisson_survival(u, v, t, self.monte_carlo_sample_size)

        self.update_embeddings(u, v, t)
        self.update_intensity_rates(output_intensity, u, v, k)
        self.update_a_and_s(u, v, k)

        return output_intensity, output_survival
    
def negative_log_likelihood(intensity_rate, survival_term):
        component_0 = torch.sum(torch.log(intensity_rate))
        component_1 = torch.sum(survival_term)
        return -component_0 + component_1

def train(model, train_data, test_data, loss_fun=negative_log_likelihood, epochs=1, batch_size=200, save_state_dict=True):
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
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
            loss = loss_fun(lambda_uvk, L_surv)
            
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            #nn.utils.clip_grad_value_(model.parameters(), 100)
            optimizer.step()

            batch_rloss[batch_idx] = loss.detach() 

            #torch.save(model.state_dict(), f"data/state_dicts/model-state-dict-batch{batch_idx+1}.pth")

            model.embeddings = model.embeddings.detach()  # reset the computational graph, no backprop over previous embedding
            model.S = model.S.detach()

            
            print(f"Batch {batch_idx+1} time:", time.time()-batch_t0)
            print(f"Batch loss: {batch_rloss[batch_idx]}")
        # end for batch

        if save_state_dict:
            torch.save(model.state_dict(), "data/state_dicts/with-time-loss-model-state-dict.pth")

        # if test_data is not None:
        #     test(model, test_data, batch_size=200)

        epoch_rloss[epoch_idx] = torch.sum(batch_rloss) / len(data_loader)
        print(f"Epoch {epoch_idx+1} time:", time.time()-epoch_t0)
        print(f"Epoch loss: {epoch_rloss[epoch_idx]}")
    #end for epoch
    return model
