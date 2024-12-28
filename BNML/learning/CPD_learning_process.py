"""
    Tests NPN
    Borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
from torch.autograd import Variable
from learning.NPN_learning_process import NPN_learning
from learning.AE_NPN_leraning_process import AE_NPN_learning
import numpy as np
import torch.optim as optim
from my_model.npn1 import rsNNC, rsNND
from data.loaddata import load_tarin_dl
import time


def CPD_train(BNML, i, para, tarin_dl_tensor_list, test_dl, features, VQVAE=None):

    VQVAE_type = None
    if para['data_file'] == 'ml-1m':
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                if BNML.list_node_label[j] == 'eDUA' or BNML.list_node_label[j] == 'eDIA':
                    VQVAE_type = True
            if i == j: # 给当前节点输出赋值
                if BNML.list_node_label[j] == 'eDUA' or BNML.list_node_label[j] == 'eDIA':
                    VQVAE_type = True
    params_list = []
    if i!= None:
        params_list.append(BNML.list_NPN[i].parameters())
    if VQVAE_type == True:
        params_list.append(VQVAE.parameters())
    optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr'], weight_decay=0) # lr
    # print(optimizer)
    # exit(0)

    best_iter = -1
    best_loss = np.inf
    convergence_iter_num = 0
    for epoch in range(1, para['BN_parameter_learning_max_iter_num'] + 1):
        # print('epoch: ' + str(epoch))
        train_loss, FBIC = CPD_learning(epoch, BNML, i, optimizer, para, tarin_dl_tensor_list, test_dl, features, VQVAE, VQVAE_type)
        # early stopping
        if epoch == 1:
            continue
        if train_loss < best_loss:
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else:
            convergence_iter_num += 1
        # print('Train Epoch: {}, best_loss: {:.6f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
        if convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num']:
            print('Train Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
            # print(output)
            break
    # exit(0)
    return FBIC



def CPD_learning(epoch, BNML, i, optimizer, para, tarin_dl_tensor_list, test_dl, features, VQVAE=None, VQVAE_type=None): # input_other = [features, adj_loop, norm, weight_A]

    train_loss_all = 0
    FBIC = 0

    for batch_idx in range(len(tarin_dl_tensor_list)): # batch_idx: 0~n
        # t1 = time.time()
        input, U_index, I_index, R = tarin_dl_tensor_list[batch_idx]
    # for batch_idx, data in enumerate(tarin_dl): # batch_idx: 0~n
    #     t1 = time.time()
    #     input, U_index, I_index, R = load_tarin_dl(data, para, BNML, input_other[0])
    #     print(input)
    #     print(input.shape)
    #     print(U_index)
    #     print(I_index)
    #     print(R)
    #     exit(0)

        data_m = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
        data_s = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
        target = torch.FloatTensor(len(input), BNML.list_cardinalities[i]).zero_().to(para['device'])

        # t2 = time.time()
        # print("time111=", "{:.4f}".format(t2 - t1))

        if para['data_file'] == 'ml-1m':
            for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                    if BNML.list_node_label[j] == 'eDUA' or BNML.list_node_label[j] == 'eDIA':
                        continue
                    else:
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] \
                            = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                if i == j: # 给当前节点输出赋值
                    if BNML.list_node_label[j] == 'eDUA' or BNML.list_node_label[j] == 'eDIA':
                        continue
                    else:
                        target = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
        else:
            for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                    data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] \
                        = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                if i == j: # 给当前节点输出赋值
                    target = input[:, sum(BNML.list_cardinalities[:i]):sum(BNML.list_cardinalities[:i+1])]
        # print(VQVAE_type)
        # print(data_m)
        # print(data_s)
        # print(target)
        # exit(0)

        ##############################################################################
        ##############################################################################
        if VQVAE_type == True: # 有与自编码相关的子节点或父节点
            train_loss, BN_loss = AE_NPN_learning(VQVAE, VQVAE_type, optimizer, epoch, para, features,
                                                   BNML, i, data_m, data_s, target, [U_index, I_index, R], test_dl)
        else:
            train_loss, BN_loss = NPN_learning(BNML, i, BNML.list_NPN[i], optimizer, epoch, para, data_m, data_s, target)
        train_loss_all += train_loss
        FBIC += BN_loss

        # print(data_m)
        # print(data_s)
        # print(target)
        # exit(0)

    # exit(0)

    return train_loss_all, FBIC * -1
