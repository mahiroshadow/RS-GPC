import torch
from torch.optim import SGD
from model import GDE
from config import *
from Dataset import CustomDataSet
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    model = GDE(user_size, item_size).to(device)
    optim = SGD(model.parameters(), lr=learning_rate)
    df = CustomDataSet()
    train = DataLoader(df, batch_size=batch_size)
    for i in range(epoch):
        for userID, itemID in train:
            userID, itemID = userID.to(device), itemID.to(device)
            loss = model(userID, itemID)
            break
            loss.backward()
            optim.step()
            optim.zero_grad()
        break
    torch.save(model, 'model.pkl')


def predict():
    pass


if __name__ == '__main__':
    # train()
    pass
