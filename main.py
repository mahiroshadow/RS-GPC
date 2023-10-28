import torch
from torch.optim import SGD
from model import GDE
from config import *
from Dataset import CustomDataSet
from torch.utils.data import DataLoader
import numpy as np
from encoder import usr_encoder, item_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    rate_matrix = torch.Tensor(np.load('./dataset/usr_item.npy')).to(device)
    model = GDE(user_size, item_size).to(device)
    optim = SGD(model.parameters(), lr=learning_rate)
    df = CustomDataSet()
    train = DataLoader(df, batch_size=batch_size)
    total_loss = 0.0
    for i in range(epoch):
        for userID, itemID in train:
            u = torch.LongTensor(usr_encoder.transform(userID)).to(device)
            p = torch.LongTensor(item_encoder.transform(itemID)).to(device)

            # 对用户交互过的物品进行采样，这里只采样1阶交互物品
            u_item_n = torch.multinomial(rate_matrix[u], 8, True)

            # 对用户交互过的物品进行采样，这里只采样1阶交互用户
            item_u_n = torch.multinomial(rate_matrix.t()[p], 8, True)

            # 负样本  size=[32]
            nega = torch.multinomial(1 - rate_matrix[u], 8, True).squeeze(1)
            loss = model(userID, p, u_item_n, item_u_n, nega)
            loss.backward()
            optim.step()
            optim.zero_grad()
            print('****************************')
            print(loss.item())
            print('****************************')
            total_loss += loss.item()
    torch.save(model, 'model.pkl')


def predict():
    pass


if __name__ == '__main__':
    train()

