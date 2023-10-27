from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CustomDataSet(Dataset):

    def __init__(self):
        self.df = pd.read_csv('./dataset/train.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        userID = np.asarray(self.df.iloc[index, 1])
        itemID = np.asarray(self.df.iloc[index, 2])
        return userID, itemID
