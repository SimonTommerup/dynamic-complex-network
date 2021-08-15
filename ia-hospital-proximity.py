import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_unique_links(batched_data):
    unique_links = torch.unique(batched_data[:,:-1], dim=0)
    return unique_links

def create_mapping(ascending_nodes):
    mapping = {}
    index = 0
    for node in ascending_nodes:
        mapping[node] = index
        index += 1
    return mapping

def map_to_zero_index(mapping, nodes):
    zero_indexed_nodes = []
    for node in nodes:
        zero_indexed_nodes.append(mapping[node])
    return zero_indexed_nodes

PATH = r"data\ia-hospital-ward-proximity.csv"

data = pd.read_csv(PATH, header=None, index_col=None, sep=",")


columns = ["u", "v", "t"]
data.columns = columns

t = [e for e in data["t"]]

tfirst = np.array(t)

first_t = t[0]
max_t = t[-1]
t = [(e-first_t)/max_t for e in t]
t = [e * 10 for e in t]

tafter = np.array(t)

print(max(t))

u = [e for e in data["u"]]
v = [e for e in data["v"]]

unique_nodes = list(set(set(u) | set(v))) # 75 unique nodes
unique_nodes.sort()

zimap = create_mapping(unique_nodes)

u = map_to_zero_index(zimap, u)
v = map_to_zero_index(zimap, v)

data_set = []
for cur_u, cur_v, cur_t in zip(u, v, t):
    if cur_u == cur_v:
        continue
    else:
        r = [cur_u, cur_v, cur_t]
        data_set.append(r)

data_set=np.array(data_set)
print("data set len", len(data_set))

time_col = 2
data_set = data_set[data_set[:,time_col].argsort()]

# verify time ordering
prev_t = 0.
for row in data_set:
    cur_t = row[time_col]
    assert cur_t >= prev_t
    prev_t = cur_t


num_train_samples = int(len(data_set)*0.80)
training_data = data_set[0:num_train_samples]
test_data = data_set[num_train_samples:]
print("Train", len(training_data))
print("test", len(test_data))
plt.hist(data_set[:,2])
plt.title("ia-hospital-ward-proximity: Event frequency")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.xlim(0.0, t[-1])
plt.vlines(x=training_data[-1][2].item(), ymin=0, ymax=15000, color="r")
plt.show()


NUM_SPLITS=200
training_batches = np.array_split(training_data, NUM_SPLITS)
print("bs=", len(training_batches[0]))

def burstiness(data):
  t, n = data[:,2], len(data[:,2])
  inter_arrivals = np.diff(t)
  mu, sigma = np.mean(inter_arrivals), np.std(inter_arrivals)
  b = (sigma - mu)/(sigma + mu)
  return b

print("B =", burstiness(data_set))

#np.save("hospital-proximity.npy", data_set)

data_set = torch.from_numpy(data_set)

unique_links = get_unique_links(data_set)