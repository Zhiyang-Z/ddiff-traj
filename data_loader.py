import torch
from torch.utils.data import Dataset
import numpy as np

class TrajectoryData(Dataset):
    def __init__(self, data_path, label_path=None):
        self.data = np.load(data_path)
        self.label = None if label_path is None else np.load(label_path)
        # self.state_num, self.len = state_num, self.data.shape[1]
        assert self.data.shape[0] == self.label.shape[0]
        self.label[:, 0] = np.floor((self.label[:, 0] % 86400) / 300)
        self.means = self.label[:,1:6].mean(axis=0)
        self.stds = self.label[:,1:6].std(axis=0)
        self.label[:,1:6] = (self.label[:,1:6] - self.means) / self.stds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
