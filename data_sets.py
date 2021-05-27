import pandas as pd
import torch
import os
from torch.utils.data import Dataset

SNAP_WIKIPEDIA = r"data/ml_wikipedia.csv"
SNAP_REDDIT = r"data/ml_reddit.csv"

class SnapDataSet(Dataset):
    def __init__(self, dataset_file_path):
        self.data_file_path = dataset_file_path
        self.data_frame = pd.read_csv(self.data_file_path)
        self.sources = [s for s in self.data_frame["u"]]
        self.destinations = [d for d in self.data_frame["i"]]
        self.timestamps = [t for t in self.data_frame["ts"]]

        self.unique_sources = self.unique(self.sources)
        self.unique_destinations = self.unique(self.destinations)
        self.n_nodes = len(self.unique(self.unique_sources | self.unique_destinations))

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, index):
        source = self.sources[index]
        destination = self.destinations[index]
        time = self.timestamps[index]
        
        return source, destination, time

    def unique(self, nodes):
        return set(nodes)


class MITDataSet(Dataset):
    def __init__(self, dataset_file_path):
        self.dataframe = pd.read_csv(dataset_file_path)
        self.sources = [s for s in self.dataframe["source"]]
        self.destinations = [d for d in self.dataframe["destination"]]
        self.times = [t for t in self.dataframe["time_delta"]]
        self.event_types = [e for e in self.dataframe["event_type"]]

        self.unique_sources = self.unique(self.sources)
        self.unique_destinations = self.unique(self.destinations)
        self.n_nodes = len(self.unique(self.unique_sources | self.unique_destinations))

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        source = self.sources[index]
        destination = self.destinations[index]
        time = self.times[index]
        event_type = self.event_types[index]
        
        return source, destination, time, event_type
    
    def unique(self, nodes):
        return set(nodes)

if __name__ == "__main__":
    wikipedia = SnapDataSet(SNAP_WIKIPEDIA)
    print(wikipedia[0])