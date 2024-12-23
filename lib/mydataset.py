from torch.utils.data import Dataset
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, days, hours, mins, target) -> None:
        days = np.load(days)
        hours = np.load(hours)
        mins = np.load(mins)
        target = np.load(target)
        self.days = torch.tensor(days, dtype=torch.float32)
        self.hours = torch.tensor(hours, dtype=torch.float32)
        self.mins = torch.tensor(mins, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)


    def __getitem__(self, index):
        return self.days[index], self.hours[index], self.mins[index], self.target[index]
    
    def __len__(self):
        return self.target.shape[0]
