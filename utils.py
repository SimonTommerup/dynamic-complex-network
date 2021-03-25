import torch
import pandas as pd

def ncat(n, tensor):
    return torch.cat(n*[tensor.view(-1)])

def read_csv(path):
    return pd.read_csv(path)


