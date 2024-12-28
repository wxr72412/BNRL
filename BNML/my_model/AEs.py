import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .nearest_embed import NearestEmbed

def save_AE(para, AE, AE_type, other = ''):
    data_path = os.getcwd()
    data_path += '\\data\\'
    # print(data_path)
    if AE_type == "VAE" or AE_type == "VGAE_AX":
        data_path += para['data_file'] + '\\' + AE_type + '-' + str(para['hidden2_dim']) + '-' + str(para['std_coefficient'])\
                     + '-adj-' + str(para['loss_adj']) + '-X-' + str(para['loss_X'])
        data_path += other
    elif AE_type == "VQVAE":
        data_path += para['data_file'] + '\\' + AE_type\
                     + '-' + str(para['hidden_dim_VQVAE_U']) + '-' + str(para['hidden_embeddings_dim_VQVAE_U']) + '-' + str(para['k_VQVAE_U'])\
                     + '-' + str(para['hidden_dim_VQVAE_I']) + '-' + str(para['hidden_embeddings_dim_VQVAE_I']) + '-' + str(para['k_VQVAE_I'])
        data_path += '-' + para['BN_file']
        data_path += other
    elif AE_type == "VQVAE":
        data_path += para['data_file'] + '\\' + AE_type\
                     + '-' + str(para['hidden_dim_VQVAE_U']) + '-' + str(para['hidden_embeddings_dim_VQVAE_U']) + '-' + str(para['k_VQVAE_U'])\
                     + '-' + str(para['hidden_dim_VQVAE_I']) + '-' + str(para['hidden_embeddings_dim_VQVAE_I']) + '-' + str(para['k_VQVAE_I'])
        data_path += '-' + para['BN_file']
        data_path += other
    elif AE_type == "VQVAE-files":
        data_path = other \
                     + 'num-' + str(para['num_VQVAE']) \
                     + '-dim-' + str(para['hidden_dim_VQVAE'][0]) \
                     + '-edim-' + str(para['hidden_embeddings_dim_VQVAE'][0]) \
                     + '-k-' + str(para['k_VQVAE'][0])
    print(data_path)
    torch.save(AE, data_path)
    print("------------Saving model is done!-------------------")

def load_AE(para, AE_type, other = ''):
    data_path = os.getcwd()
    data_path += '\\data\\'
    if AE_type == 'VAE' or 'AE_type' == 'VGAE_AX':
        data_path += para['data_file'] + '\\' + AE_type + '-' + str(para['hidden2_dim']) + '-' + str(para['std_coefficient']) + '-adj-' + str(para['loss_adj']) + '-X-' + str(para['loss_X'])
        data_path += other
    elif AE_type == 'VQVAE':
        data_path += para['data_file'] + '\\' + AE_type \
                     + '-' + str(para['hidden_dim_VQVAE_U']) + '-' + str(para['hidden_embeddings_dim_VQVAE_U']) + '-' + str(para['k_VQVAE_U']) \
                     + '-' + str(para['hidden_dim_VQVAE_I']) + '-' + str(para['hidden_embeddings_dim_VQVAE_I']) + '-' + str(para['k_VQVAE_I'])
        data_path += '-' + para['BN_file']
        data_path += other
    elif AE_type == "VQVAE-files":
        data_path = other \
                     + 'num-' + str(para['num_VQVAE']) \
                     + '-dim-' + str(para['hidden_dim_VQVAE'][0]) \
                     + '-edim-' + str(para['hidden_embeddings_dim_VQVAE'][0]) \
                     + '-k-' + str(para['k_VQVAE'][0])
    print(data_path)
    return torch.load(data_path)

def load_VQVAE(para, str = ''):
    data_path = os.getcwd()
    data_path += '\\data\\'
    data_path += para['data_file'] + '\\' + str
    print(data_path)
    return torch.load(data_path)

class VQVAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""
    def __init__(self, para, features, hidden=10, k=2, vq_coef=0.1, comit_coef=0.2):
        super(VQVAE, self).__init__()
        self.para = para
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0
        self.features = features


        if self.para['data_file'] == 'ml-1m' or self.para['data_file'] == 'dermatology':
            self.hidden_U = self.para['hidden_dim_VQVAE_U']
            self.embeddings_dim_U = para['hidden_embeddings_dim_VQVAE_U']
            self.num_embeddings_U = self.para['k_VQVAE_U']
            self.hidden_I = self.para['hidden_dim_VQVAE_I']
            self.embeddings_dim_I = para['hidden_embeddings_dim_VQVAE_I']
            self.num_embeddings_I = self.para['k_VQVAE_I']

            self.indexU = None
            self.indexI = None

            self.embU = NearestEmbed(self.num_embeddings_U, self.embeddings_dim_U)
            self.base_encodeU = nn.Sequential(
                nn.Linear(features[0].shape[1], 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, self.hidden_U),
                nn.LeakyReLU(),
            )
            self.decoderU = nn.Sequential(
                nn.Linear(self.hidden_U, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(),
                nn.Linear(128, features[0].shape[1]),
                nn.Sigmoid(),  # compress to a range (0, 1)
            )

            self.embI = NearestEmbed(self.num_embeddings_I, self.embeddings_dim_I)
            self.base_encodeI = nn.Sequential(
                nn.Linear(features[1].shape[1], 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, self.hidden_I),
                nn.LeakyReLU(),
            )
            self.decoderI = nn.Sequential(
                nn.Linear(self.hidden_I, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, features[1].shape[1]),
                nn.Sigmoid(),  # compress to a range (0, 1)
            )
        elif self.para['data_file'] == 'bone-marrow':
            self.hidden_U = self.para['hidden_dim_VQVAE_U']
            self.embeddings_dim_U = para['hidden_embeddings_dim_VQVAE_U']
            self.num_embeddings_U = self.para['k_VQVAE_U']
            self.hidden_I = self.para['hidden_dim_VQVAE_I']
            self.embeddings_dim_I = para['hidden_embeddings_dim_VQVAE_I']
            self.num_embeddings_I = self.para['k_VQVAE_I']

            self.indexU = None
            self.indexI = None

            self.embU = NearestEmbed(self.num_embeddings_U, self.embeddings_dim_U)
            self.base_encodeU = nn.Sequential(
                nn.Linear(features[0].shape[1], 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, self.hidden_U),
                nn.LeakyReLU(),
            )
            self.decoderU = nn.Sequential(
                nn.Linear(self.hidden_U, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, features[0].shape[1]),
                nn.Sigmoid(),  # compress to a range (0, 1)
            )

            self.embI = NearestEmbed(self.num_embeddings_I, self.embeddings_dim_I)
            self.base_encodeI = nn.Sequential(
                nn.Linear(features[1].shape[1], 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, self.hidden_I),
                nn.LeakyReLU(),
            )
            self.decoderI = nn.Sequential(
                nn.Linear(self.hidden_I, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, features[1].shape[1]),
                nn.Sigmoid(),  # compress to a range (0, 1)
            )
        elif self.para['data_file'] == 'toy2' or \
                self.para['data_file'] == 'child' or self.para['data_file'] == 'pigs' or self.para['data_file'] == 'water' or self.para['data_file'] == 'munin1':
            self.list_hidden = self.para['hidden_dim_VQVAE']
            self.list_embeddings_dim = para['hidden_embeddings_dim_VQVAE']
            self.list_num_embeddings = self.para['k_VQVAE']

            self.list_embed = [NearestEmbed(self.list_num_embeddings[i], self.list_embeddings_dim[i]).to(para['device']) for i in range(self.para['num_VQVAE'])]
            self.list_base_encode = [nn.Sequential(
                nn.Linear(features[i].shape[1], 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, self.list_hidden[i]),
                nn.LeakyReLU(),).to(para['device']) for i in range(self.para['num_VQVAE'])]
            self.list_decoder = [nn.Sequential(
                nn.Linear(self.list_hidden[i], 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, features[i].shape[1]),
                nn.Sigmoid(),).to(para['device']) for i in range(self.para['num_VQVAE'])]
        else:
            self.hidden = self.para['hidden_dim_VQVAE']
            self.embeddings_dim = para['hidden_embeddings_dim_VQVAE']
            self.k = self.para['k_VQVAE']
            self.num_embeddings = self.k
            self.emb = NearestEmbed(self.num_embeddings, self.embeddings_dim)
            self.base_encode = nn.Sequential(
                nn.Linear(features.shape[1], 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, self.hidden),
                nn.LeakyReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, features.shape[1]),
                nn.Sigmoid(),  # compress to a range (0, 1)
            )

    def encode(self, x, i = None):
        if self.para['data_file'] == 'ml-1m' or self.para['data_file'] == 'dermatology' or self.para['data_file'] == 'bone-marrow':
            h1_U = self.base_encodeU(x[0])
            h2_U = h1_U.view(-1, self.embeddings_dim_U, int(self.hidden_U / self.embeddings_dim_U))
            h1_I = self.base_encodeI(x[1])
            h2_I = h1_I.view(-1, self.embeddings_dim_I, int(self.hidden_I / self.embeddings_dim_I))
            return [h2_U, h2_I]
        elif self.para['data_file'] == 'toy2' or \
                self.para['data_file'] == 'child' or self.para['data_file'] == 'pigs' or self.para['data_file'] == 'water' or self.para['data_file'] == 'munin1':
            # print(x[i])
            h1 = self.list_base_encode[i](x[i])
            # print(h1)
            h2 = h1.view(-1, self.list_embeddings_dim[i], int(self.list_hidden[i] / self.list_embeddings_dim[i]))
            # print(h2)
            # exit(0)
            return h2
        else:
            h1 = self.base_encode(x)
            # print(h1.shape)
            h2 = h1.view(-1, self.embeddings_dim, int(self.hidden / self.embeddings_dim))
            # print(h2.shape)
            # exit(0)
            return h2

    def decode(self, z, i = None):
        if self.para['data_file'] == 'ml-1m' or self.para['data_file'] == 'dermatology' or self.para['data_file'] == 'bone-marrow':
            X_pred_U = self.decoderU(z[0].view(-1, self.hidden_U))
            X_pred_I = self.decoderI(z[1].view(-1, self.hidden_I))
            return [X_pred_U, X_pred_I]
        elif self.para['data_file'] == 'toy2' or \
                self.para['data_file'] == 'child' or self.para['data_file'] == 'pigs' or self.para['data_file'] == 'water' or self.para['data_file'] == 'munin1':
            X_pred = self.list_decoder[i](z.view(-1, self.list_hidden[i]))
            return X_pred
        else:
            X_pred = self.decoder(z.view(-1, self.hidden))
            return X_pred

    def forward(self, x, VQVAE_index = None):
        if self.para['data_file'] == 'ml-1m' or self.para['data_file'] == 'dermatology' or self.para['data_file'] == 'bone-marrow':
            print(x)
            z_e = self.encode(x)
            # print(z_e[0].shape)
            # print(z_e[1].shape)
            # exit(0)
            (self.z_q_U, self.indexU) = self.embU(z_e[0], weight_sg=True)
            (emb_U, _) = self.embU(z_e[0].detach())
            (self.z_q_I, self.indexI) = self.embI(z_e[1], weight_sg=True)
            (emb_I, _) = self.embI(z_e[1].detach())

            # print(z_q_U.shape)
            # print(z_q_I.shape)
            # print(index1_U.shape)
            # print(index1_I.shape)
            # exit(0)
            X_pred = self.decode([self.z_q_U, self.z_q_I])
            # print(X_pred[0].shape)
            # print(X_pred[1].shape)
            # exit(0)
            return [self.indexU, self.indexI], z_e, [self.z_q_U, self.z_q_I], [emb_U, emb_I], X_pred
        elif self.para['data_file'] == 'toy2' or \
                self.para['data_file'] == 'child' or self.para['data_file'] == 'pigs' or self.para['data_file'] == 'water' or self.para['data_file'] == 'munin1':
            # print(x)
            self.list_z_e = []
            self.list_index = []
            self.list_z_q = []
            self.list_emb = []
            self.list_X_pred = []
            # print(VQVAE_index)
            # exit(0)
            for i in range(self.para['num_VQVAE']):
                if VQVAE_index == None or VQVAE_index == i:
                    z_e = self.encode(x, i)
                    # print(z_e)
                    # print(z_e.shape)
                    self.list_z_e.append(z_e)
                    # print(len(self.list_z_e))
                    # exit(0)
                    (z_q, index) = self.list_embed[i](self.list_z_e[i], weight_sg=True)
                    (emb, _) = self.list_embed[i](self.list_z_e[i].detach())
                    # print(z_q)
                    # print(z_q.shape)
                    # print(emb)
                    # print(z_q.shape)
                    # print(index)
                    # print(index.shape)
                    # exit(0)
                    self.list_index.append(index)
                    self.list_z_q.append(z_q)
                    self.list_emb.append(emb)
                    X_pred = self.decode(self.list_z_q[i], i)
                    self.list_X_pred.append(X_pred)
                else:
                    self.list_z_e.append(None)
                    self.list_index.append(None)
                    self.list_z_q.append(None)
                    self.list_emb.append(None)
                    self.list_X_pred.append(None)
            # print(X_pred[0])
            # print(X_pred[0].shape)
            # print(X_pred[1].shape)
            # exit(0)
            return self.list_index, self.list_z_e, self.list_z_q, self.list_emb, self.list_X_pred
        else:
            z_e = self.encode(x)
            # z_e[0][0][0] = 1.0
            # z_e[0][1][0] = 0.3
            # print(z_e[0])
            # print(z_e.shape)
            # exit(0)
            (z_q, index1) = self.emb(z_e, weight_sg=True)
            (emb, index2) = self.emb(z_e.detach())
            # print(self.emb.weight)
            # print(z_q[0])
            # print(emb[0])
            # print(index1[0])
            # print(index2[0])
            # exit(0)

            # print(self.emb.weight.shape)
            # print(z_q.shape)
            # print(emb.shape)
            # print(index1.shape)
            # print(index2.shape)
            # exit(0)
            X_pred = self.decode(z_q)
            # return X_pred, z_e, emb
            return index1, z_e, z_q, emb, X_pred



    # def sample(self, size):
    #     sample = torch.randn(size, self.emb_size,
    #                          int(self.hidden / self.emb_size))
    #     if self.cuda():
    #         sample = sample.cuda()
    #     emb, _ = self.emb(sample)
    #     sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
    #     return sample

    def loss(self, X_pred, features, z_e, emb, i = None):
        if self.para['data_file'] == 'ml-1m' or self.para['data_file'] == 'dermatology':
            self.ce_lossU = F.binary_cross_entropy(X_pred[0].reshape(-1), features[0].reshape(-1))
            # self.ce_lossU = F.mse_loss(X_pred[0].reshape(-1), features[0].reshape(-1))
            self.vq_lossU = F.mse_loss(emb[0].reshape(-1), z_e[0].detach().reshape(-1))
            self.commit_lossU = F.mse_loss(z_e[0].reshape(-1), emb[0].detach().reshape(-1))

            self.ce_lossI = F.binary_cross_entropy(X_pred[1].reshape(-1), features[1].reshape(-1))
            # self.ce_lossI = F.mse_loss(X_pred[1].reshape(-1), features[1].reshape(-1))
            self.vq_lossI = F.mse_loss(emb[1].reshape(-1), z_e[1].detach().reshape(-1))
            self.commit_lossI = F.mse_loss(z_e[1].reshape(-1), emb[1].detach().reshape(-1))

            self.d_weightU = 0
            for i in range(self.num_embeddings_U-1):
                self.d_weightU += F.mse_loss(self.embU.weight[:,i], self.embU.weight[:,i+1])
                # print(self.d_weightU)
            self.d_weightI = 0
            for i in range(self.num_embeddings_I-1):
                self.d_weightI += F.mse_loss(self.embI.weight[:,i], self.embI.weight[:,i+1])
                # print(self.d_weightI)

            # self.L2 = (z_e[0] ** 2 + emb[0] ** 2).sum(1).mean() + (z_e[1] ** 2 + emb[1] ** 2).sum(1).mean()
            # print(self.ce_lossU)
            # print(self.vq_lossU)
            # print(self.commit_lossU)
            # print(self.ce_lossI)
            # print(self.vq_lossI)
            # print(self.commit_lossI)
            # print()
            #
            # print(features[0].shape[1]) # torch.Size([6040, 30]) -- 30
            # print(features[1].shape[1]) # torch.Size([3645, 119])) -- 119
            # exit(0)
            return (self.ce_lossU + self.vq_coef*self.vq_lossU + self.comit_coef*self.commit_lossU) \
                   + (self.ce_lossI + self.vq_coef*self.vq_lossI + self.comit_coef*self.commit_lossI) \
                   + self.d_weightU * 6e-7 \
                   + self.d_weightI * 4e-7
                   # + self.L2

            # return self.ce_lossU + self.vq_coef*self.vq_lossU + self.comit_coef*self.commit_lossU
            # return self.ce_lossI + self.vq_coef*self.vq_lossI + self.comit_coef*self.commit_lossI

        elif self.para['data_file'] == 'toy2' or \
                self.para['data_file'] == 'child' or self.para['data_file'] == 'pigs' or self.para['data_file'] == 'water' or self.para['data_file'] == 'munin1':
            # print(features)
            # print(X_pred)
            # exit(0)
            loss = 0
            if i == None:
                for i in range(self.para['num_VQVAE']):
                    ce_loss = F.binary_cross_entropy(X_pred[i].reshape(-1), features[i].reshape(-1))
                    # self.ce_lossU = F.mse_loss(X_pred[i].reshape(-1), features[i].reshape(-1))
                    vq_loss = F.mse_loss(emb[i].reshape(-1), z_e[i].detach().reshape(-1))
                    commit_loss = F.mse_loss(z_e[i].reshape(-1), emb[i].detach().reshape(-1))

                    # d_weight = 0
                    # for j in range(self.list_num_embeddings[i]-1):
                    #     d_weight += F.mse_loss(self.list_emb[i].weight[:,j], self.list_emb[i].weight[:,j+1])
                        # print(self.d_weight)

                    # loss += self.vq_coef * vq_loss + self.comit_coef * commit_loss
                    loss += ce_loss + self.vq_coef * vq_loss + self.comit_coef * commit_loss
                    # loss += ce_loss + self.vq_coef * vq_loss + self.comit_coef * commit_loss + self.d_weight * self.para['d_weight']
            else:
                # print(i)
                # exit(0)
                ce_loss = F.binary_cross_entropy(X_pred[i].reshape(-1), features[i].reshape(-1))
                vq_loss = F.mse_loss(emb[i].reshape(-1), z_e[i].detach().reshape(-1))
                commit_loss = F.mse_loss(z_e[i].reshape(-1), emb[i].detach().reshape(-1))

                # loss += self.vq_coef * vq_loss + self.comit_coef * commit_loss
                loss += ce_loss + self.vq_coef * vq_loss + self.comit_coef * commit_loss
            return loss

        elif self.para['data_file'] == 'bone-marrow':
            self.ce_lossU = F.binary_cross_entropy(X_pred[0].reshape(-1), features[0].reshape(-1))
            # self.ce_lossU = F.mse_loss(X_pred[0].reshape(-1), features[0].reshape(-1))
            self.vq_lossU = F.mse_loss(emb[0].reshape(-1), z_e[0].detach().reshape(-1))
            self.commit_lossU = F.mse_loss(z_e[0].reshape(-1), emb[0].detach().reshape(-1))

            # print(X_pred[0].shape)
            # print(features[0].shape)
            #
            # print(X_pred[0][0])
            # print(features[0][0])

            self.ce_lossI = F.binary_cross_entropy(X_pred[1].reshape(-1), features[1].reshape(-1))
            # self.ce_lossI = F.mse_loss(X_pred[1].reshape(-1), features[1].reshape(-1))
            self.vq_lossI = F.mse_loss(emb[1].reshape(-1), z_e[1].detach().reshape(-1))
            self.commit_lossI = F.mse_loss(z_e[1].reshape(-1), emb[1].detach().reshape(-1))

            # print(X_pred[1].shape)
            # print(features[1].shape)
            #
            # print(X_pred[1][0])
            # print(features[1][0])
            # print()

            self.d_weightU = 0
            for i in range(self.num_embeddings_U-1):
                self.d_weightU += F.mse_loss(self.embU.weight[:,i], self.embU.weight[:,i+1])
                # print(self.d_weightU)
            self.d_weightI = 0
            for i in range(self.num_embeddings_I-1):
                self.d_weightI += F.mse_loss(self.embI.weight[:,i], self.embI.weight[:,i+1])
                # print(self.d_weightI)

            # self.L2 = (z_e[0] ** 2 + emb[0] ** 2).sum(1).mean() + (z_e[1] ** 2 + emb[1] ** 2).sum(1).mean()
            # print(self.ce_lossU)
            # print(self.vq_lossU)
            # print(self.commit_lossU)
            # print(self.ce_lossI)
            # print(self.vq_lossI)
            # print(self.commit_lossI)
            # print()
            #
            # print(features[0].shape[1]) # torch.Size([6040, 30]) -- 30
            # print(features[1].shape[1]) # torch.Size([3645, 119])) -- 119
            # exit(0)
            return (self.ce_lossU + self.vq_coef*self.vq_lossU + self.comit_coef*self.commit_lossU) \
                   + (self.ce_lossI + self.vq_coef*self.vq_lossI + self.comit_coef*self.commit_lossI) \
                   + self.d_weightU * 6e-7 \
                   + self.d_weightI * 4e-7
                   # + self.L2
        else:
            # print(x)
            # print(recon_x)
            # print(z_e)
            # print(emb)
            # exit(0)
            self.ce_loss = F.binary_cross_entropy(X_pred.reshape(-1), features.reshape(-1)) # fc1.weight.grad fc2.weight.grad fc3.weight.grad fc4.weight.grad
            # self.ce_loss = F.mse_loss(recon_x.reshape(-1), x.reshape(-1))
            self.vq_loss = F.mse_loss(emb.reshape(-1), z_e.detach().reshape(-1)) # emb.weight.grad
            self.commit_loss = F.mse_loss(z_e.reshape(-1), emb.detach().reshape(-1)) # fc1.weight.grad fc2.weight.grad
            return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    # def latest_losses(self):
    #     return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}



# class VQVAE(nn.Module):
#     """Vector Quantized AutoEncoder for mnist"""
#
#     def __init__(self, para, features, hidden=10, k=2, vq_coef=0.1, comit_coef=0.2):
#         super(VQVAE, self).__init__()
#         self.para = para
#
#         self.uEmbd = nn.Embedding(features[0].shape[0], 64).to(para['device'])
#         self.iEmbd = nn.Embedding(features[1].shape[0], 64).to(para['device'])
#
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.vq_coef = vq_coef
#         self.comit_coef = comit_coef
#         self.ce_loss = 0
#         self.vq_loss = 0
#         self.commit_loss = 0
#         # self.fc1 = nn.Linear(784, 400)
#         # self.fc2 = nn.Linear(400, hidden)
#         # self.fc3 = nn.Linear(hidden, 400)
#         # self.fc4 = nn.Linear(400, 784)
#
#
#         if self.para['data_file'] == 'ml-1m':
#             self.hidden_U = self.para['hidden_dim_VQVAE_U']
#             self.embeddings_dim_U = para['hidden_embeddings_dim_VQVAE_U']
#             self.num_embeddings_U = self.para['k_VQVAE_U']
#             self.hidden_I = self.para['hidden_dim_VQVAE_I']
#             self.embeddings_dim_I = para['hidden_embeddings_dim_VQVAE_I']
#             self.num_embeddings_I = self.para['k_VQVAE_I']
#
#             self.indexU = None
#             self.indexI = None
#
#             self.embU = NearestEmbed(self.num_embeddings_U, self.embeddings_dim_U)
#             self.base_encodeU = nn.Sequential(
#                 nn.Linear(64, 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, 128),
#                 nn.LeakyReLU(),
#                 nn.Linear(128, self.hidden_U),
#                 nn.LeakyReLU(),
#             )
#             self.decoderU = nn.Sequential(
#                 nn.Linear(self.hidden_U, 128),
#                 nn.LeakyReLU(),
#                 nn.Linear(128, 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, features[0].shape[1]),
#                 nn.Sigmoid(),  # compress to a range (0, 1)
#             )
#
#             self.embI = NearestEmbed(self.num_embeddings_I, self.embeddings_dim_I)
#             self.base_encodeI = nn.Sequential(
#                 nn.Linear(64, 512),
#                 nn.LeakyReLU(),
#                 nn.Linear(512, 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, self.hidden_I),
#                 nn.LeakyReLU(),
#             )
#             self.decoderI = nn.Sequential(
#                 nn.Linear(self.hidden_I, 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, 512),
#                 nn.LeakyReLU(),
#                 nn.Linear(512, features[1].shape[1]),
#                 nn.Sigmoid(),  # compress to a range (0, 1)
#             )
#         else:
#             self.hidden = self.para['hidden_dim_VQVAE']
#             self.embeddings_dim = para['hidden_embeddings_dim_VQVAE']
#             self.k = self.para['k_VQVAE']
#             self.num_embeddings = self.k
#             self.emb = NearestEmbed(self.num_embeddings, self.embeddings_dim)
#             self.base_encode = nn.Sequential(
#                 nn.Linear(features.shape[1], 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, 128),
#                 nn.LeakyReLU(),
#                 nn.Linear(128, self.hidden),
#                 nn.LeakyReLU(),
#             )
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.hidden, 128),
#                 nn.LeakyReLU(),
#                 nn.Linear(128, 256),
#                 nn.LeakyReLU(),
#                 nn.Linear(256, features.shape[1]),
#                 nn.Sigmoid(),  # compress to a range (0, 1)
#             )
#
#     def encode(self, x):
#         if self.para['data_file'] == 'ml-1m':
#             h1_U = self.base_encodeU(x[0])
#             h2_U = h1_U.view(-1, self.embeddings_dim_U, int(self.hidden_U / self.embeddings_dim_U))
#             h1_I = self.base_encodeI(x[1])
#             h2_I = h1_I.view(-1, self.embeddings_dim_I, int(self.hidden_I / self.embeddings_dim_I))
#             return [h2_U, h2_I]
#         else:
#             h1 = self.base_encode(x)
#             # print(h1.shape)
#             h2 = h1.view(-1, self.embeddings_dim, int(self.hidden / self.embeddings_dim))
#             # print(h2.shape)
#             # exit(0)
#             return h2
#
#     def decode(self, z):
#         if self.para['data_file'] == 'ml-1m':
#             X_pred_U = self.decoderU(z[0].view(-1, self.hidden_U))
#             X_pred_I = self.decoderI(z[1].view(-1, self.hidden_I))
#             return [X_pred_U, X_pred_I]
#         else:
#             X_pred = self.decoder(z.view(-1, self.hidden))
#             return X_pred
#
#     def forward(self, x):
#         if self.para['data_file'] == 'ml-1m':
#             index_U = torch.LongTensor([i for i in range(self.uEmbd.num_embeddings)]).to(self.para['device'])
#             uembd = self.uEmbd(index_U)
#             index_I = torch.LongTensor([i for i in range(self.iEmbd.num_embeddings)]).to(self.para['device'])
#             iembd = self.iEmbd(index_I)
#             x = [uembd, iembd]
#             z_e = self.encode(x)
#             # print(z_e[0].shape)
#             # print(z_e[1].shape)
#             # exit(0)
#             (z_q_U, index1_U) = self.embU(z_e[0], weight_sg=True)
#             (emb_U, index2_U) = self.embU(z_e[0].detach())
#             (z_q_I, index1_I) = self.embI(z_e[1], weight_sg=True)
#             (emb_I, index2_I) = self.embI(z_e[1].detach())
#
#             self.indexU = index1_U
#             self.indexI = index1_I
#             # print(z_q_U.shape)
#             # print(z_q_I.shape)
#             # print(index1_U.shape)
#             # print(index1_I.shape)
#             # exit(0)
#             X_pred = self.decode([z_q_U, z_q_I])
#             # print(X_pred[0].shape)
#             # print(X_pred[1].shape)
#             # exit(0)
#             return [index1_U, index1_I], z_e, [z_q_U, z_q_I], [emb_U, emb_I], X_pred
#         else:
#             z_e = self.encode(x)
#             # z_e[0][0][0] = 1.0
#             # z_e[0][1][0] = 0.3
#             # print(z_e[0])
#             # print(z_e.shape)
#             # exit(0)
#             (z_q, index1) = self.emb(z_e, weight_sg=True)
#             (emb, index2) = self.emb(z_e.detach())
#             # print(self.emb.weight)
#             # print(z_q[0])
#             # print(emb[0])
#             # print(index1[0])
#             # print(index2[0])
#             # exit(0)
#
#             # print(self.emb.weight.shape)
#             # print(z_q.shape)
#             # print(emb.shape)
#             # print(index1.shape)
#             # print(index2.shape)
#             # exit(0)
#             X_pred = self.decode(z_q)
#             # return X_pred, z_e, emb
#             return index1, z_e, z_q, emb, X_pred
#
#
#
#     # def sample(self, size):
#     #     sample = torch.randn(size, self.emb_size,
#     #                          int(self.hidden / self.emb_size))
#     #     if self.cuda():
#     #         sample = sample.cuda()
#     #     emb, _ = self.emb(sample)
#     #     sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
#     #     return sample
#
#     def loss(self, X_pred, features, z_e, emb):
#         if self.para['data_file'] == 'ml-1m':
#             self.ce_lossU = F.binary_cross_entropy(X_pred[0].reshape(-1), features[0].reshape(-1))
#             # self.ce_lossU = F.mse_loss(X_pred[0].reshape(-1), features[0].reshape(-1))
#             self.vq_lossU = F.mse_loss(emb[0].reshape(-1), z_e[0].detach().reshape(-1))
#             self.commit_lossU = F.mse_loss(z_e[0].reshape(-1), emb[0].detach().reshape(-1))
#
#             self.ce_lossI = F.binary_cross_entropy(X_pred[1].reshape(-1), features[1].reshape(-1))
#             # self.ce_lossI = F.mse_loss(X_pred[1].reshape(-1), features[1].reshape(-1))
#             self.vq_lossI = F.mse_loss(emb[1].reshape(-1), z_e[1].detach().reshape(-1))
#             self.commit_lossI = F.mse_loss(z_e[1].reshape(-1), emb[1].detach().reshape(-1))
#
#             # self.L2 = (z_e[0] ** 2 + emb[0] ** 2).sum(1).mean() + (z_e[1] ** 2 + emb[1] ** 2).sum(1).mean()
#             # print(self.ce_lossU)
#             # print(self.vq_lossU)
#             # print(self.commit_lossU)
#             # print(self.ce_lossI)
#             # print(self.vq_lossI)
#             # print(self.commit_lossI)
#             # print()
#
#             return self.ce_lossU + self.vq_coef*self.vq_lossU + self.comit_coef*self.commit_lossU \
#                    + self.ce_lossI + self.vq_coef*self.vq_lossI + self.comit_coef*self.commit_lossI \
#                 # + self.L2 * 0.001
#             # return self.ce_lossU + self.vq_coef*self.vq_lossU + self.comit_coef*self.commit_lossU
#             # return self.ce_lossI + self.vq_coef*self.vq_lossI + self.comit_coef*self.commit_lossI
#         else:
#             # print(x)
#             # print(recon_x)
#             # print(z_e)
#             # print(emb)
#             # exit(0)
#             self.ce_loss = F.binary_cross_entropy(X_pred.reshape(-1), features.reshape(-1)) # fc1.weight.grad fc2.weight.grad fc3.weight.grad fc4.weight.grad
#             # self.ce_loss = F.mse_loss(recon_x.reshape(-1), x.reshape(-1))
#             self.vq_loss = F.mse_loss(emb.reshape(-1), z_e.detach().reshape(-1)) # emb.weight.grad
#             self.commit_loss = F.mse_loss(z_e.reshape(-1), emb.detach().reshape(-1)) # fc1.weight.grad fc2.weight.grad
#             return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss
#
#     # def latest_losses(self):
#     #     return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}