import numpy as np
import pandas as pd
import os
import torch
from encoder import usr_encoder, item_encoder
import gc

# 用户数/物品数(测试方法如下)
'''
df = pd.read_csv('./dataset/artists.dat',
                 sep='\t',
                 names=['id', 'name', 'url', 'pictureURL'],
                 skiprows=1)
item = set()
for i in df['id']:
    item.add(i)
print(len(item))
'''
user_size, item_size = 1892, 17632

smooth_ratio = 0.1
# rough_ratio = 0.0


def cal_spectral_feature(Adj, size, type='user', largest=True, niter=5):
    value, vector = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)
    np.save(f'./dataset/{type}_smooth_value.npy', value.cpu().numpy())
    np.save(f'./dataset/{type}_smooth_feature.npy', vector.cpu().numpy())


# 划分数据集训练集
def train_test_split(df, split_ratio=0.8):
    # 判断是否已经划分过了
    if not os.path.exists('./dataset/train.csv'):
        df_size = len(df)
        train_size = int(df_size * split_ratio)
        # test_size = df_size - train_size
        df_train = df.iloc[0:train_size].sample(frac=1,
                                                replace=False,
                                                random_state=999)
        df_train.reset_index(inplace=True, drop=True)
        df_train.to_csv('./dataset/train.csv')
        df_test = df.iloc[train_size:df_size].sample(frac=1,
                                                     replace=False,
                                                     random_state=999)
        df_test.reset_index(inplace=True, drop=True)
        df_test.to_csv('./dataset/test.csv')


def func():

    # 读取用户-用户交互数据
    # df_usr_friend=pd.read_csv('./dataset/user_friends.dat',sep='\t',names=['userID','friendID'],skiprows=1)
    # 读取用户物品-物品交互数据
    df = pd.read_csv('./dataset/user_artists.dat',
                     sep='\t',
                     names=['userID', 'itemID', 'weight'],
                     skiprows=1)
    # df_item=pd.read_csv('./dataset/artists.dat',sep='\t',names=['id','name','url','pictureURL'],skiprows=1)

    usr_item_matrix = torch.zeros(user_size, item_size).cuda()
    for row in df.itertuples():
        x, y = usr_encoder.transform([row[1]
                                      ])[0], item_encoder.transform([row[2]
                                                                     ])[0]
        usr_item_matrix[x, y] = 1
    # np.save('./dataset/usr_item.npy', usr_item_matrix.cpu().numpy())
    degree_u = usr_item_matrix.sum(1)
    degree_i = usr_item_matrix.sum(0)
    for i in range(user_size):
        if degree_u[i] != 0:
            degree_u[i] = 1 / degree_u[i].sqrt()

    for i in range(item_size):
        if degree_i[i] != 0:
            degree_i[i] = 1 / degree_i[i].sqrt()
    usr_item_matrix = degree_u.unsqueeze(1) * usr_item_matrix * degree_i

    # 清除gpu缓存

    del degree_i, degree_u
    gc.collect()
    torch.cuda.empty_cache()
    '''
    L_i = usr_item_matrix.t().mm(usr_item_matrix)
    cal_spectral_feature(L_i,
                         size=int(0.3 * item_size),
                         type='item',
                         largest=True)
    del L_i
    gc.collect()
    torch.cuda.empty_cache()
    '''
    df_usr_friend = pd.read_csv('./dataset/user_friends.dat',
                                sep='\t',
                                names=['userID', 'friendID'],
                                skiprows=1)
    usr_usr_matrix = torch.zeros(user_size, user_size)

    for row in df_usr_friend.itertuples():
        x, y = usr_encoder.transform([row[1]
                                      ])[0], usr_encoder.transform([row[1]])[0]
        usr_usr_matrix[x, y] = 1

    degree_u = usr_usr_matrix.sum(1)
    degree_f = usr_usr_matrix.sum(0)
    for i in range(user_size):
        if degree_u[i] != 0:
            degree_u[i] = 1 / degree_u[i].sqrt()
        if degree_f[i] != 0:
            degree_f[i] = 1 / degree_f[i].sqrt()
    usr_usr_matrix = degree_u.unsqueeze(1) * usr_usr_matrix * degree_f
    cal_spectral_feature(usr_usr_matrix,
                         size=int(user_size * smooth_ratio),
                         largest=True)


if __name__ == '__main__':
    func()