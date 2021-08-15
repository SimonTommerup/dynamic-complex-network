#%%

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_unique_links(batched_data):
    unique_links = torch.unique(batched_data[:,:-1], dim=0)
    return unique_links


#%%
path = r"data\ia-radoslaw-email.csv"

columns = ["u", "v", "type", "t"]
data = pd.read_csv(path, header=None, index_col=None, delim_whitespace=True)
data.columns = columns

t = [e for e in data["t"]]
first_t = t[0]
max_t = t[-1]
t = [(e-first_t)/max_t for e in t]
t = [e * 1000 for e in t]
print(min(t))
print(max(t))
u = [e-1 for e in data["u"]]
v = [e-1 for e in data["v"]]


unique_nodes = set(set(u) | set(v))
print(len(unique_nodes))


# plt.hist(t, bins=20)
# plt.show()

equal_count = 0
data_set = []
for cur_u, cur_v, cur_t in zip(u, v, t):
  if cur_u == cur_v:
    continue
  else:
    r = [cur_u, cur_v, cur_t]
    data_set.append(r)


data_set=np.array(data_set)

num_train_samples = int(len(data_set)*0.80)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]
print("Train", len(training_data))
print("test", len(test_data))
plt.hist(data_set[:,2])
plt.title("ia-radoslaw-email: Event frequency")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.xlim(0.0, t[-1])
plt.vlines(x=training_data[-1][2].item(), ymin=0, ymax=15000, color="r")
plt.show()


u2 = data_set[:,0]
v2 = data_set[:,1]

unique_nodes2 = set(set(u) | set(v))
print(len(unique_nodes2))


def burstiness(data):
  t, n = data[:,2], len(data[:,2])
  inter_arrivals = np.diff(t)
  mu, sigma = np.mean(inter_arrivals), np.std(inter_arrivals)
  b = (sigma - mu)/(sigma + mu)
  return b

print("B =", burstiness(test_data)) 

#np.save("radoslaw-email.npy", data_set)



# %%
