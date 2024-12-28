import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
loss_func_MSE = nn.MSELoss()
np.set_printoptions(threshold=np.inf)


def AE_learning(VQVAE, VQVAE_type, optimizer, epoch, para, features):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    # t0 = time.time()
    train_loss = 0
    optimizer.zero_grad()
    # t1 = time.time()
    # print("time0-1=", "{:.4f}".format(t1 - t0))
    if VQVAE_type == True:
        VQVAE.train()
        Z, z_e, z_q, emb, X_pred = VQVAE(features)
        # print(Z3)
        # print(z_e[0])
        # print(emb[0])
        # print(X_pred3)
        # exit(0)
    # t2 = time.time()
    # print("time1-2=", "{:.4f}".format(t2 - t1))
    ####################################################################################################################
    ####################################################################################################################
    # t3 = time.time()
    # print("time2-3=", "{:.4f}".format(t3 - t2))
    if VQVAE_type == True:
        # print(X_pred3)
        # print(features)
        # print(z_e)
        # print(emb)
        # exit(0)
        VQVAE_loss = VQVAE.loss(X_pred, features, z_e, emb)
        # print('VQVAE_loss: ')
        # print(VQVAE_loss)
        train_loss += VQVAE_loss

    train_loss.backward()
    optimizer.step()
    ####################################################################################################################
    ####################################################################################################################
    # t6 = time.time()
    # print("time5-6=", "{:.4f}".format(t6 - t5))

    # if 1:
    if epoch % para['print_interval'] == 0:
        if para['data_file'] == 'ml-1m':
            if VQVAE_type == True:
                print('VQVAE.embU')
                print(VQVAE.embU.weight)
                print('VQVAE.embI')
                print(VQVAE.embI.weight)

                print('self.d_weightU')
                d_embU = torch.FloatTensor(VQVAE.num_embeddings_U, VQVAE.num_embeddings_U).zero_()
                for i in range(VQVAE.num_embeddings_U):
                    for j in range(VQVAE.num_embeddings_U):
                        d_embU[i][j] = F.mse_loss(VQVAE.embU.weight[:,i], VQVAE.embU.weight[:,j])
                print(d_embU)

                print('self.d_weightI')
                d_embI = torch.FloatTensor(VQVAE.num_embeddings_I, VQVAE.num_embeddings_I).zero_()
                for i in range(VQVAE.num_embeddings_I):
                    for j in range(VQVAE.num_embeddings_I):
                        d_embI[i][j] = F.mse_loss(VQVAE.embI.weight[:,i], VQVAE.embI.weight[:,j])
                print(d_embI)
                # exit(0)

                # print(Z3)
                print("epoch:", '%d' % (epoch),
                      "train_loss:", '%6f' % (train_loss),
                      "BCE_X_U=", "{:.6f}".format(F.binary_cross_entropy(X_pred[0].reshape(-1), features[0].reshape(-1), weight=None)),
                      "BCE_X_I", "{:.6f}".format(F.binary_cross_entropy(X_pred[1].reshape(-1), features[1].reshape(-1), weight=None)),
                      "MSE_X_U=", "{:.6f}".format(loss_func_MSE(X_pred[0].reshape(-1), features[0].reshape(-1))),
                      "MSE_X_I=", "{:.6f}".format(loss_func_MSE(X_pred[1].reshape(-1), features[1].reshape(-1))),
                      )
        else:
            if VQVAE_type == True:
                    # print(Z)
                    print("epoch:", '%d' % (epoch),
                          "train_loss:", '%6f' % (train_loss),
                          "BCE_X=", "{:.6f}".format(F.binary_cross_entropy(X_pred.reshape(-1), features.reshape(-1), weight=None)),
                          "MSE_X=", "{:.6f}".format(loss_func_MSE(X_pred.reshape(-1), features.reshape(-1)))
                          )
    return train_loss.item(), Z