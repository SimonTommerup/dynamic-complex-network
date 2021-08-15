import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch

def burstiness(data):
  t, n = data[:,2], len(data[:,2])
  inter_arrivals = np.diff(t)
  mu, sigma = np.mean(inter_arrivals), np.std(inter_arrivals)
  b = (sigma - mu)/(sigma + mu)
  return b

def get_unique_links(batched_data):
    unique_links = torch.unique(batched_data[:,:-1], dim=0)
    return unique_links

PATH = r"data\ia-contact.csv"

data = pd.read_csv(PATH, header=None, index_col=None, delim_whitespace=True)

columns = ["u", "v", "discard", "t"]
data.columns = columns

u = [elem-1 for elem in data["u"]]
v = [elem-1 for elem in data["v"]]
t = [elem for elem in data["t"]]

unique_nodes = list(set(set(u) | set(v))) # 274 unique nodes
print("unique nodes", len(unique_nodes))

data_set = []
for cur_u, cur_v, cur_t in zip(u, v, t):
    if cur_u == cur_v:
        continue
    else:
        r = [cur_u, cur_v, cur_t]
        data_set.append(r)

data_set = np.array(data_set, dtype=np.double)
u2 = data_set[:,0]
v2 = data_set[:,1]


unique_nodes = list(set(set(data_set[:,0]) | set(data_set[:,1]))) # 274 unique nodes
print("Unique nodes after removing nodes: ", len(unique_nodes))

time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]
data_set[:,2] = (data_set[:,2] - data_set[0,2]) / data_set[-1,2]

# verify time ordering
prev_t = 0.
for idx, row in enumerate(data_set):
    cur_t = row[time_col]
    assert cur_t >= prev_t, print(idx)
    prev_t = cur_t

print("B =", burstiness(data_set)) 

# discard last 11 observations.
data_set = data_set[:-11]
#plt.hist(data_set[:,2],bins=100)
#plt.show()
#plt.close()

num_train_samples = int(len(data_set)*0.80)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]
print("Train", len(training_data))
print("test", len(test_data))
NUM_SPLITS=170
training_batches = np.array_split(training_data, NUM_SPLITS)
print("bs=", len(training_batches[0]))


plt.hist(data_set[:,2],bins=10)
plt.title("ia-contact: Event frequency")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.xlim(0.0, data_set[-1,2])
plt.vlines(x=training_data[-1][2].item(), ymin=0, ymax=8000, color="r")
plt.show()
plt.close()

# data_set = torch.from_numpy(data_set)

# unique_links = get_unique_links(data_set)
# print(len(unique_links))
#print("Fraction: ", len(unique_links) / (274*273/2))


#np.save("contact.npy", data_set)



