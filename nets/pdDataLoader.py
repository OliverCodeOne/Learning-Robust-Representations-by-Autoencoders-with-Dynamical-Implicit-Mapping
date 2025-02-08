import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop("classify", axis=1).values
        self.labels = dataframe["classify"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = torch.Tensor(self.features[index])
        label = self.labels[index]

        return feature, label
