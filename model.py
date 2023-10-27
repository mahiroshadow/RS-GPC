import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc

rate_matrix = torch.Tensor(np.load('./dataset/usr_item.npy')).cuda()


class GDE(nn.Module):

    def __init__(self,
                 user_size,
                 item_size,
                 beta=5.0,
                 feature_type='smoothed',
                 drop_out=0.1,
                 latent_size=64,
                 reg=0.01):
        super(GDE, self).__init__()

        # 对用户和物品进行嵌入(dim=64)
        self.user_embed = torch.nn.Embedding(user_size, latent_size)
        self.item_embed = torch.nn.Embedding(item_size, latent_size)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

        self.beta = beta
        self.reg = reg
        self.drop_out = drop_out
        if drop_out != 0:
            self.m = torch.nn.Dropout(drop_out)

        if feature_type == 'smoothed':
            user_filter = self.weight_feature(
                torch.Tensor(
                    np.load('./dataset/user_smooth_value.npy')).cuda())

            user_vector = torch.Tensor(
                np.load('./dataset/user_smooth_feature.npy')).cuda()

        self.L_u = (user_vector * user_filter).mm(user_vector.t())

        del user_vector, user_filter
        gc.collect()
        torch.cuda.empty_cache()

    def weight_feature(self, value):
        return torch.exp(self.beta * value)

    def forward(self, user, pos_item, nega_item, loss_type='adaptive'):

        if self.drop_out == 0:
            final_user, final_pos, final_nega = self.L_u[user].mm(
                self.user_embed.weight), self.L_i[pos_item].mm(
                    self.item_embed.weight), self.L_i[nega_item].mm(
                        self.item_embed.weight)

        else:
            final_user,final_pos,final_nega=(self.m(self.L_u[user])*(1-self.drop_out)).mm(self.user_embed.weight),(self.m(self.L_i[pos_item])*(1-self.drop_out)).mm(self.item_embed.weight),\
            (self.m(self.L_i[nega_item])*(1-self.drop_out)).mm(self.item_embed.weight)

        if loss_type == 'adaptive':

            res_nega = (final_user * final_nega).sum(1)
            nega_weight = (
                1 - (1 - res_nega.sigmoid().clamp(max=0.99)).log10()).detach()
            out = ((final_user * final_pos).sum(1) -
                   nega_weight * res_nega).sigmoid()

        else:
            out = ((final_user * final_pos).sum(1) -
                   (final_user * final_nega).sum(1)).sigmoid()

        reg_term = self.reg * (final_user**2 + final_pos**2 +
                               final_nega**2).sum()
        return (-torch.log(out).sum() + reg_term) / batch_size

    def predict_matrix(self):

        final_user = self.L_u.mm(self.user_embed.weight)
        final_item = self.L_i.mm(self.item_embed.weight)
        #mask the observed interactions
        return (final_user.mm(final_item.t())).sigmoid() - rate_matrix * 1000

    def test(self):
        #calculate idcg@k(k={1,...,20})
        def cal_idcg(k=20):
            idcg_set = [0]
            scores = 0.0
            for i in range(1, k + 1):
                scores += 1 / np.log2(1 + i)
                idcg_set.append(scores)

            return idcg_set

        def cal_score(topn, now_user, trunc=20):
            dcg10, dcg20, hit10, hit20 = 0.0, 0.0, 0.0, 0.0
            for k in range(trunc):
                max_item = topn[k]
                if test_data[now_user].count(max_item) != 0:
                    if k <= 10:
                        dcg10 += 1 / np.log2(2 + k)
                        hit10 += 1
                    dcg20 += 1 / np.log2(2 + k)
                    hit20 += 1

            return dcg10, dcg20, hit10, hit20

        #accuracy on test data
        ndcg10, ndcg20, recall10, recall20 = 0.0, 0.0, 0.0, 0.0

        final_user = self.L_u.mm(self.user_embed.weight)
        predict = self.predict_matrix()

        idcg_set = cal_idcg()
        for now_user in range(user_size):
            test_lens = len(test_data[now_user])

            #number of test items truncated at k
            all10 = 10 if (test_lens > 10) else test_lens
            all20 = 20 if (test_lens > 20) else test_lens

            #calculate dcg
            topn = predict[now_user].topk(20)[1]

            dcg10, dcg20, hit10, hit20 = cal_score(topn, now_user)

            ndcg10 += (dcg10 / idcg_set[all10])
            ndcg20 += (dcg20 / idcg_set[all20])
            recall10 += (hit10 / all10)
            recall20 += (hit20 / all20)

        ndcg10, ndcg20, recall10, recall20 = round(
            ndcg10 / user_size,
            4), round(ndcg20 / user_size,
                      4), round(recall10 / user_size,
                                4), round(recall20 / user_size, 4)
        print(ndcg10, ndcg20, recall10, recall20)

        result.append([ndcg10, ndcg20, recall10, recall20])
