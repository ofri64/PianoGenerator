import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        # Load data and get label
        X = torch.tensor(self.x_list[index])
        y = torch.tensor(self.y_list[index])

        return X, y
