import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from func.metrics import get_scores
import time
from data.loaddata import load_tarin_dl
from func.EM_D import filling_Data
import torch.optim as optim


loss_func_MSE = nn.MSELoss()
np.set_printoptions(threshold=np.inf)

# from torchviz import make_dot, make_dot_from_trace


def All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features, VQVAE, VQVAE_type):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    # t0 = time.time()



    params_list = []
    for i in range(0, BNML.num_nodes):
        params_list.append(BNML.list_NPN[i].parameters())
    if VQVAE_type == True:
        params_list.append(VQVAE.parameters())
    # print(len(params_list))
    # exit(0)
    optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr'], weight_decay=0) # lr





    best_iter = -1
    best_loss = np.inf
    convergence_iter_num = 0
    for epoch in range(1, para['BN_parameter_learning_max_iter_num'] + 1):

        train_loss_all = 0
        BNML.list_BIC_loglikelihood = [float(0) for i in range(0, BNML.num_nodes)] # 存放个变量的对数似然
        for batch_idx in range(len(tarin_dl_tensor_list)): # batch_idx: 0~n
            input, U_index, I_index, R = tarin_dl_tensor_list[batch_idx]
            # print(input)
            # print(U_index)
            # print(I_index)
            # print(R)
            # exit(0)
            train_loss = 0 # grad_fn=<AddBackward0>

            if VQVAE != None:
                VQVAE.train()
                Z, z_e, z_q, emb, X_pred = VQVAE(features)

            # for i in range(1):
            #     i = 12
            for i in range(BNML.num_nodes):
                BNML.list_NPN[i].train()

                data_m = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
                data_s = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
                target = torch.FloatTensor(len(input), BNML.list_cardinalities[i]).zero_().to(para['device'])
                if para['data_file'] == 'ml-1m':
                    for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                        if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                            if BNML.list_node_label[j] == 'R':
                                data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] \
                                    = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                        if i == j: # 给当前节点输出赋值
                            if BNML.list_node_label[j] == 'R':
                                target = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                else:
                    for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                        if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                            data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] \
                                = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                        if i == j: # 给当前节点输出赋值
                            target = input[:, sum(BNML.list_cardinalities[:i]):sum(BNML.list_cardinalities[:i+1])]


                if para['data_file']  == 'ml-1m':
                    for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                        if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                            # if BNML.list_node_label[j] == 'eDUA':
                            #     index = 0
                            #     for k in range(BNML.num_nodes):
                            #         if BNML.list_node_label[k] == 'eDUA':
                            #             if k == j:
                            #                 break
                            #             if k != j:
                            #                 index += 1
                            #     data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
                            #         torch.FloatTensor(VQVAE.indexU.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexU[:, index].unsqueeze(1), 1)[U_index]
                            # elif BNML.list_node_label[j] == 'eDIA':
                            #     index = 0
                            #     for k in range(BNML.num_nodes):
                            #         if BNML.list_node_label[k] == 'eDIA':
                            #             if k == j:
                            #                 break
                            #             if k != j:
                            #                 index += 1
                            #     data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
                            #         torch.FloatTensor(VQVAE.indexI.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexI[:, index].unsqueeze(1), 1)[I_index]
                            if BNML.list_node_label[j] == 'eDUA':
                                index = 0
                                for k in range(BNML.num_nodes):
                                    if BNML.list_node_label[k] == 'eDUA':
                                        if k == j:
                                            break
                                        if k != j:
                                            index += 1
                                data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[0][U_index][...,index]
                                # print(data_m[0]) # torch.Size([8])
                                # exit(0)
                            elif BNML.list_node_label[j] == 'eDIA':
                                index = 0
                                for k in range(BNML.num_nodes):
                                    if BNML.list_node_label[k] == 'eDIA':
                                        if k == j:
                                            break
                                        if k != j:
                                            index += 1
                                data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[1][I_index][...,index]
                        if i == j: # 给当前节点输出赋值
                            # if BNML.list_node_label[j] == 'eDUA':
                            #     index = 0
                            #     for k in range(BNML.num_nodes):
                            #         if BNML.list_node_label[k] == 'eDUA':
                            #             if k == j:
                            #                 break
                            #             if k != j:
                            #                 index += 1
                            #     target = torch.FloatTensor(VQVAE.indexU.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexU[:, index].unsqueeze(1), 1)[U_index]
                            # elif BNML.list_node_label[j] == 'eDIA':
                            #     index = 0
                            #     for k in range(BNML.num_nodes):
                            #         if BNML.list_node_label[k] == 'eDIA':
                            #             if k == j:
                            #                 break
                            #             if k != j:
                            #                 index += 1
                            #     target = torch.FloatTensor(VQVAE.indexI.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexI[:, index].unsqueeze(1), 1)[I_index]
                            if BNML.list_node_label[j] == 'eDUA':
                                index = 0
                                for k in range(BNML.num_nodes):
                                    if BNML.list_node_label[k] == 'eDUA':
                                        if k == j:
                                            break
                                        if k != j:
                                            index += 1
                                target = z_q[0][U_index][...,index]
                                # print(data_m[0]) # torch.Size([8])
                                # exit(0)
                            elif BNML.list_node_label[j] == 'eDIA':
                                index = 0
                                for k in range(BNML.num_nodes):
                                    if BNML.list_node_label[k] == 'eDIA':
                                        if k == j:
                                            break
                                        if k != j:
                                            index += 1
                                target = z_q[1][I_index][...,index]

                # data_m = torch.FloatTensor(2, BNML.sum_cardinalities).zero_().to(para['device'])
                # data_s = torch.FloatTensor(2, BNML.sum_cardinalities).zero_().to(para['device'])
                # target = torch.FloatTensor(2, BNML.list_cardinalities[i]).zero_().to(para['device'])
                # data_m[0][0] = 1
                # data_m[1][0] = 1
                #
                # target[0][0] = 1
                # target[1][0] = 2
                # target[0][1] = 1
                # target[1][1] = 3

                # print(data_m)
                # print(data_m.shape)
                # print(data_s)
                # print(data_s.shape)
                # print(target)
                # print(target.shape)
                # print(prior)
                # print(output)
                #
                # print(VQVAE.indexU)
                # print(U_index)
                # print('VQVAE.embU')
                # print(VQVAE.embU.weight)
                #
                # print(VQVAE.indexI)
                # print(I_index)
                # print(R)
                # print(VQVAE.indexI[8])
                # print(VQVAE.indexI[9])
                # print('VQVAE.embI')
                # print(VQVAE.embI.weight)
                # print()
                # exit(0)



                ##################################################################################
                ##################################################################################
                w = None
                list_latent = []
                for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                    if BNML.edges[j][i] == 1 or i == j: # 父节点或子节点j
                        if BNML.list_latent_variables[j] == 1:
                            list_latent.append(j)
                # print(1)
                if list_latent != []:
                    data_m, data_s, target, w = filling_Data(BNML, i, list_latent, data_m, target, para, [U_index, I_index, R], VQVAE)
                # print(2)
                # print(data_m)
                # print(data_m[0])
                # print(data_m[2])
                # print(data_m[1])
                # print(data_m[3])
                # print(data_s)
                # print(target)
                # print(target[0])
                # print(w)
                # print(data_m.shape)
                # print(data_s.shape)
                # print(target.shape)
                # print(w.shape)
                # w = torch.FloatTensor(data_m.shape[0], 1).zero_().to(para['device'])
                # w[0] = 0.7
                # w[1] = 0.3
                # exit(0)
                ##################################################################################
                ##################################################################################
                prior = None
                # prior = torch.FloatTensor(1, 8).zero_().to(para['device'])
                # prior[0][0] = 0
                # prior[0][1] = 0
                ##################################################################################
                ##################################################################################
                BN_loss = BNML.list_NPN[i].loss(data_m, data_s, target, w, prior)
                BNML.list_BIC_loglikelihood[i] += BN_loss.item() * -1
                train_loss += BN_loss * para['BN_loss_ratio']

                if epoch % para['print_interval'] == 0:
                    print('Node: {}, Train Epoch: {}, BN loss: {:.4f}, Batch size: {}, EM Batch size: {}\n'.format(i+1, epoch, BN_loss, len(input), len(data_m)))
                # exit(0)
            # print(BNML.list_BIC_loglikelihood)
            # print(train_loss)
            VQVAE_loss = 0
            if VQVAE_type == True:
                VQVAE_loss = VQVAE.loss(X_pred, features, z_e, emb)
                train_loss += VQVAE_loss
            # print(VQVAE_loss)
            # print(train_loss)

            optimizer.zero_grad()  # 清空过往梯度
            train_loss.backward() # 反向传播，计算当前梯度
            optimizer.step() # 根据梯度更新网络参数
            train_loss_all += train_loss.item()
            print(batch_idx)
            # exit(0)

        # if 1:
        if epoch % para['print_interval'] == 0:
            print('Epoch: {}, Train loss: {:.4f}'.format(epoch, train_loss_all))
            print(BNML.list_BIC_loglikelihood)
            if VQVAE != None:
                if para['data_file'] == 'ml-1m':
                    print("epoch:", '%d' % (epoch),
                          "VQVAE_loss:", '%6f' % (VQVAE_loss),
                          "BCE_X_U=", "{:.6f}".format(F.binary_cross_entropy(X_pred[0].reshape(-1), features[0].reshape(-1), weight=None)),
                          "BCE_X_I", "{:.6f}".format(F.binary_cross_entropy(X_pred[1].reshape(-1), features[1].reshape(-1), weight=None)),
                          "MSE_X_U=", "{:.6f}".format(loss_func_MSE(X_pred[0].reshape(-1), features[0].reshape(-1))),
                          "MSE_X_I=", "{:.6f}".format(loss_func_MSE(X_pred[1].reshape(-1), features[1].reshape(-1))))
                else:
                    print("epoch:", '%d' % (epoch),
                          "VQVAE_loss:", '%6f' % (VQVAE_loss),
                          "BCE_X=", "{:.6f}".format(F.binary_cross_entropy(X_pred.reshape(-1), features.reshape(-1), weight=None)),
                          "MSE_X=", "{:.6f}".format(loss_func_MSE(X_pred.reshape(-1), features.reshape(-1))))

        # print(train_loss_all)
        # print(epoch)
        # early stopping
        if train_loss_all < best_loss:
            best_loss = train_loss_all
            best_iter = epoch
            convergence_iter_num = 0
        else:
            convergence_iter_num += 1
        if convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num']:
            print('Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
            return BNML, VQVAE