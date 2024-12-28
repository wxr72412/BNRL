"""
    Tests NPN
    Borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import para_init as para_init
# from torchviz import make_dot, make_dot_from_trace


def NPN_MAP_process(BNML, para, test_dl_tensor_list):
    # for i in range(BNML.num_nodes):
    #     BNML.list_NPN[i].eval() # my_model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，my_model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

    # for batch_idx in range(len(test_dl_tensor_list)):
    input, _, _, _ = test_dl_tensor_list[0]
    # print(input) # [12, 2, 2, 3, 3, 3, 3, 2, 5, 5, 2, 2, 4, 3, 3, 3, 2, 3]
    # print(U_index)
    # print(I_index)
    # print(R)
    # exit(0)

    list_not_evidence_var = [] # 将非证据变量作为待优化的参数
    list_not_evidence_var_Norm = [] # 归一化后的待优化的参数
    list_not_evidence_var_index = [None for i in range(0, BNML.num_nodes)] # 将非证据变量对应参数的index
    index = 0

    for j in range(BNML.num_nodes): # 将证据变量的值变为one-hot，并生成非证据变量的参数集合以便传入优化器中
        not_evidence_var = None
        if j+1 in BNML.list_evidence_node:
            continue
        else:
            list_not_evidence_var_index[j] = index
            index += 1
            # print(BNML.list_node_type[j])
            if BNML.list_node_type[j] == 'C':
                not_evidence_var = Variable(torch.FloatTensor(len(input), BNML.list_cardinalities[j]).zero_()).to(para['device'])
            elif BNML.list_node_type[j] == 'D':
                not_evidence_var = Variable(torch.FloatTensor(len(input), BNML.list_cardinalities[j]).zero_() + 0.5).to(para['device'])
                # print(not_evidence_var)
                # exit(0)
            list_not_evidence_var.append(nn.Parameter(not_evidence_var))
            list_not_evidence_var_Norm.append(None)
            # print(list_not_evidence_var[j])
            # print(list_not_evidence_var_Norm[j])
            # exit(0)
    # print('list_not_evidence_var:')
    # print(list_not_evidence_var)
    print('list_not_evidence_var_index:')
    print(list_not_evidence_var_index)
    print('list_not_evidence_var_Norm:')
    print(list_not_evidence_var_Norm)
    # exit(0)

    optimizer_MAP = optim.Adam(list_not_evidence_var, lr = 0.01) # lr
    # optimizer_MAP = optim.ASGD(list_not_evidence_var, lr=0.01)  # lr
    data_m_temp = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])

    ##########################################   先将与证据变量相邻的变量更新取值   #########################################
    ##########################################   先将与证据变量相邻的变量更新取值   #########################################
    print('##########################################   先将与证据变量相邻的变量更新取值   #########################################')
    list_not_evidence_var_index_nearEvidenceVar = []
    for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
        for e in BNML.list_evidence_node:  # 0 ~ num_nodes-1
            if BNML.edges[j][e-1] == 1 or BNML.edges[e-1][j] == 1 :
                if j not in list_not_evidence_var_index_nearEvidenceVar:
                    list_not_evidence_var_index_nearEvidenceVar.append(j)
    print('list_not_evidence_var_index_nearEvidenceVar')
    print(list_not_evidence_var_index_nearEvidenceVar)
    if para_init.bn == 'child':
        list_not_evidence_var_index_nearEvidenceVar = [11-3]
    if para_init.bn == 'water':
        list_not_evidence_var_index_nearEvidenceVar = [23-3, 24-3]
    if para_init.bn == 'pigs':
        # list_not_evidence_var_index_nearEvidenceVar.append(260-3)
        # list_not_evidence_var_index_nearEvidenceVar.append(327-3)
        list_not_evidence_var_index_nearEvidenceVar = [45 - 3, 46 - 3]
    if para_init.bn == 'munin1':
        # list_not_evidence_var_index_nearEvidenceVar.append(91 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(92 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(93 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(94 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(95 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(100 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(101 - 3)
        # list_not_evidence_var_index_nearEvidenceVar.append(102 - 3)
        list_not_evidence_var_index_nearEvidenceVar = [111 - 3, 112 - 3]

    print('list_not_evidence_var_index_nearEvidenceVar')
    print(list_not_evidence_var_index_nearEvidenceVar)

    best_iter = -1
    best_loss = np.inf
    convergence_iter_num = 0
    t_start = time.time()
    for epoch in range(1, para['MAP_max_iter_num'] + 1):
        # for epoch in range(1, 3):
        #     print(epoch)
        train_loss = 0
        AddBackward_loss = 0  # grad_fn=<AddBackward0>
        for i in list_not_evidence_var_index_nearEvidenceVar:
            data_m = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
            data_s = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
            target = torch.FloatTensor(len(input), BNML.list_cardinalities[i]).zero_().to(para['device'])
            for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
                # print(data[:, i]) # 第i列
                if BNML.edges[j][i] == 1:  # j是i的父节点
                    if j + 1 in BNML.list_evidence_node:  # j是证据变量，赋值父节点j对应的列
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = input[:,
                                                                                                           sum(BNML.list_cardinalities[
                                                                                                               :j]):sum(
                                                                                                               BNML.list_cardinalities[
                                                                                                               :j + 1])]
                    else:
                        if BNML.list_node_type[j] == 'C':
                            data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = \
                            list_not_evidence_var[list_not_evidence_var_index[j]]
                        elif BNML.list_node_type[j] == 'D':
                            if para['MAP_Norm'] == 'Softmax':
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = F.softmax(
                                    list_not_evidence_var[list_not_evidence_var_index[j]],
                                    dim=1)  # 对每一行进行softmax归一化为0到1的值，且和为1
                                # temo_exp = torch.exp(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp)
                            elif para['MAP_Norm'] == 'Sigmoid':
                                temo_exp = torch.sigmoid(list_not_evidence_var[list_not_evidence_var_index[j]])  # 对每一行进行softmax归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp, dim=1).unsqueeze(1)
                            elif para['MAP_Norm'] == 'Relu':
                                temp = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]])  # 对每一行进行归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temp / torch.sum(temp, dim=1).unsqueeze(1)
                            # data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = list_not_evidence_var_Norm[list_not_evidence_var_index[j]]
                            data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]])
                if i == j:  # 给当前节点输出赋值
                    if j + 1 in BNML.list_evidence_node:
                        target = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])]
                    else:
                        if BNML.list_node_type[j] == 'C':
                            target = list_not_evidence_var[list_not_evidence_var_index[j]]
                        elif BNML.list_node_type[j] == 'D':
                            if para['MAP_Norm'] == 'Softmax':
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = F.softmax(list_not_evidence_var[list_not_evidence_var_index[j]], dim=1)  # 对每一行进行softmax归一化为0到1的值，且和为1
                                # temo_exp = torch.exp(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp)
                            elif para['MAP_Norm'] == 'Sigmoid':
                                temo_exp = torch.sigmoid(list_not_evidence_var[list_not_evidence_var_index[j]])  # 对每一行进行softmax归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp, dim=1).unsqueeze(1)
                            elif para['MAP_Norm'] == 'Relu':
                                temp = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]])  # 对每一行进行归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temp / torch.sum(temp, dim=1).unsqueeze(1)
                            target = list_not_evidence_var_Norm[list_not_evidence_var_index[j]]
            # print(i)
            # print(data_m) # [12, 2, 2, 3, 3, 3, 3, 2, 5, 5, 2, 2, 4, 3, 3, 3, 2, 3]
            # print(data_s[0])
            # print(target)
            # print( BNML.list_NPN[i]((data_m, data_s))[0] )
            # print(list_not_evidence_var_index[i])
            # if list_not_evidence_var_index[i] != None:
            #     print(list_not_evidence_var_Norm[list_not_evidence_var_index[i]])
            # exit(0)
            AddBackward_loss += BNML.list_NPN[i].loss(data_m, data_s, target, w = None, prior = None, print_output = 0)
        optimizer_MAP.zero_grad()  # 清空过往梯度
        AddBackward_loss.backward()  # 反向传播，计算当前梯度
        optimizer_MAP.step()  # 根据梯度更新网络参数
        train_loss += AddBackward_loss.item()

        # print('x:')
        # print(x)
        # print('list_not_evidence_var:')
        # print(list_not_evidence_var)
        # print('list_not_evidence_var_Norm:')
        # print(list_not_evidence_var_Norm)
        # print('target:')
        # print(target)
        # exit(0)

        # # if 1:
        if epoch % para['print_interval'] == 0:
            print('MAP Epoch: {}, Train loss: {:.4f}'.format(epoch, train_loss))
            # print('list_not_evidence_var:')
            # print(list_not_evidence_var)
            # print('list_not_evidence_var_Norm:')
            # print(list_not_evidence_var_Norm)

        # early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else:
            convergence_iter_num += 1
        if convergence_iter_num == para['MAP_max_convergence_iter_num'] or epoch == para['MAP_max_iter_num']:
            print('MAP Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
            # print(list_not_evidence_var)
            # print(list_not_evidence_var_Norm)
            # print(list_not_evidence_var_index)
            # exit(0)
            print("MAP epoch time=", "{:.4f}".format(time.time() - t_start))

            return list_not_evidence_var, list_not_evidence_var_Norm, list_not_evidence_var_index
    # print('list_not_evidence_var:')
    # print(list_not_evidence_var)
    print('list_not_evidence_var_Norm:')
    print(list_not_evidence_var_Norm)
    # exit(0)
    ##########################################   先将与证据变量相邻的变量更新取值   #########################################
    ##########################################   先将与证据变量相邻的变量更新取值   #########################################
    print('##########################################   将与证据变量 不 相邻的变量更新取值   #########################################')



    best_iter = -1
    best_loss = np.inf
    convergence_iter_num = 0
    t_start = time.time()
    for epoch in range(1, para['MAP_max_iter_num'] + 1):
    # for epoch in range(1, 3):
    #     print(epoch)
        train_loss = 0
        AddBackward_loss = 0 # grad_fn=<AddBackward0>
        for i in range(BNML.num_nodes):
            if i in list_not_evidence_var_index_nearEvidenceVar:
                continue
        # for i in range(5, 6):
        # for i in range(1, 2):
        #     data_m = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
        #     data_s = Variable(torch.zeros(data_m.size())).to(para['device'])
            data_m = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
            data_s = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改

            target = torch.FloatTensor(len(input), BNML.list_cardinalities[i]).zero_().to(para['device'])
            # print(data_m)
            # print(data_s)
            # print(target)
            # print()
            # exit(0)
            for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                # print(data[:, i]) # 第i列
                if BNML.edges[j][i] == 1: # j是i的父节点
                    if j+1 in BNML.list_evidence_node: # j是证据变量，赋值父节点j对应的列
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                    else:
                        if BNML.list_node_type[j] == 'C':
                            data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = list_not_evidence_var[list_not_evidence_var_index[j]]
                        elif BNML.list_node_type[j] == 'D':
                            if para['MAP_Norm'] == 'Softmax':
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = F.softmax(list_not_evidence_var[list_not_evidence_var_index[j]], dim = 1) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # temo_exp = torch.exp(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp)
                            elif para['MAP_Norm'] == 'Sigmoid':
                                temo_exp = torch.sigmoid(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp, dim=1).unsqueeze(1)
                            elif para['MAP_Norm'] == 'Relu':
                                temp = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temp / torch.sum(temp, dim=1).unsqueeze(1)
                            # data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = list_not_evidence_var_Norm[list_not_evidence_var_index[j]]
                            data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]])
                if i == j: # 给当前节点输出赋值
                    if j+1 in BNML.list_evidence_node:
                        target = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
                    else:
                        if BNML.list_node_type[j] == 'C':
                            target = list_not_evidence_var[list_not_evidence_var_index[j]]
                        elif BNML.list_node_type[j] == 'D':
                            if para['MAP_Norm'] == 'Softmax':
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = F.softmax(list_not_evidence_var[list_not_evidence_var_index[j]], dim = 1) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # temo_exp = torch.exp(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                # list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp)
                            elif para['MAP_Norm'] == 'Sigmoid':
                                temo_exp = torch.sigmoid(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行softmax归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temo_exp / torch.sum(temo_exp, dim=1).unsqueeze(1)
                            elif para['MAP_Norm'] == 'Relu':
                                temp = torch.relu(list_not_evidence_var[list_not_evidence_var_index[j]]) # 对每一行进行归一化为0到1的值，且和为1
                                list_not_evidence_var_Norm[list_not_evidence_var_index[j]] = temp / torch.sum(temp, dim=1).unsqueeze(1)
                            target = list_not_evidence_var_Norm[list_not_evidence_var_index[j]]

                            # print(list_not_evidence_var[list_not_evidence_var_index[j]])
                            # print(temo_exp)
                            # print(target)
                            # exit(0)
            # print(i)
            # print(data_m[0])
            # print(data_s[0])
            # print(target[0])
            # print(list_not_evidence_var_index[i])
            # if list_not_evidence_var_index[i] != None:
            #     print(list_not_evidence_var_Norm[list_not_evidence_var_index[i]])
            # exit(0)

            print_output = 0
            # if epoch % para['print_interval'] == 0:
            # # if 1:
            #     print_output = 1
            #     print('Train Epoch: {} node: {}'.format(epoch, i))
            #     print('data_m:')
            #     print(data_m)
            #     # print(data_m.shape)
            #     print('list_not_evidence_var:')
            #     print(list_not_evidence_var)
            #     # print(data_m[:, 0:2])
            #     # print(data_m[:, 2:4])
            #     # print('data_s:')
            #     # print(data_s)
            #     # print(data_s.shape)
            #     print('target:')
            #     print(target)
            #     # print()
            #     # exit(0)
            w = None
            prior = None
            AddBackward_loss += BNML.list_NPN[i].loss(data_m, data_s, target, w, prior, print_output)
            # print(AddBackward_loss)

        # print('list_not_evidence_var:')
        # print(list_not_evidence_var)
        # print(target)
        optimizer_MAP.zero_grad()  # 清空过往梯度
        AddBackward_loss.backward() # 反向传播，计算当前梯度
        optimizer_MAP.step() # 根据梯度更新网络参数
        train_loss += AddBackward_loss.item()

        # print('x:')
        # print(x)
        # print('list_not_evidence_var:')
        # print(list_not_evidence_var)
        # print('list_not_evidence_var_Norm:')
        # print(list_not_evidence_var_Norm)
        # print('target:')
        # print(target)
        # exit(0)

        if 1:
        # if epoch % para['print_interval'] == 0:
            print('MAP Epoch: {}, Train loss: {:.4f}'.format(epoch, train_loss))
            # print('list_not_evidence_var:')
            # print(list_not_evidence_var)
            # print('list_not_evidence_var_Norm:')
            # print(list_not_evidence_var_Norm)

        # early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else:
            convergence_iter_num += 1
        if convergence_iter_num == para['MAP_max_convergence_iter_num'] or epoch == para['MAP_max_iter_num']:
            print('MAP Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
            print('list_not_evidence_var:')
            print(list_not_evidence_var)
            print('list_not_evidence_var_Norm:')
            print(list_not_evidence_var_Norm)
            # print(list_not_evidence_var_index)
            # exit(0)
            print("MAP epoch time=", "{:.4f}".format(time.time() - t_start))

            return list_not_evidence_var, list_not_evidence_var_Norm, list_not_evidence_var_index


