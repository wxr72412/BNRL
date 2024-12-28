import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from func.metrics import get_scores
import time
from data.loaddata import load_tarin_dl
from func.EM_toy import filling_Data
import torch.optim as optim
from my_model.npn1 import GaussianNPN, vanillaNN
from func.find_cpd import find_cpt


loss_func_MSE = nn.MSELoss()
np.set_printoptions(threshold=np.inf)

# from torchviz import make_dot, make_dot_from_trace


def All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features, df, VQVAE, VQVAE_type, net, curr_iter):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    # t0 = time.time()

    if VQVAE_type == False:
        w_temp = None
        for i in range(0, BNML.num_nodes):
            # print(i)
            # if i+1 != 19: # CHILD  L=2 最后一个变量
            #     continue

            if curr_iter == 0 and BNML.list_dependent_latent_node(i) != []: # 第0次迭代为预训练，仅学习与隐变量无关的显变量的CPD
                continue
            # else:
            #     print("pre: " + str(i+1) )
            if curr_iter > 1 and BNML.list_dependent_latent_node(i) == []: # 除了第一次迭代外，其他时候不学习与隐变量无关的显变量的CPD
            # if curr_iter > 0 and BNML.list_dependent_latent_node(i) == []:  # 除了第一次迭代外，其他时候不学习与隐变量无关的显变量的CPD
            # if BNML.list_dependent_latent_node(i) == []:
                continue
            # i = 14 - 1
            # if i+1 > 4:
            #     break
            #######################################################################################

            params_list = []
            params_list.append(BNML.list_NPN[i].parameters())
            optimizer = optim.Adam([{'params': p} for p in params_list], lr=para['lr'], weight_decay=0)  # lr
            #######################################################################################

            # 隐变量的值填充后，数据量过大，导致data_s = torch.FloatTensor(len(data_m), BNML.sum_cardinalities).zero_().to(para['device'])操作耗时
            max_index_tarin_dl_tensor_list = len(tarin_dl_tensor_list)
            last_batch_size = len(tarin_dl_tensor_list[max_index_tarin_dl_tensor_list - 1][0])
            # print(max_index_tarin_dl_tensor_list)
            # print(last_batch_size)

            # print(para['batch_size'])
            # print(last_batch_size)
            # exit(0)
            # data_m_temp = torch.FloatTensor(para['batch_size'], BNML.sum_cardinalities).zero_().to(para['device'])
            data_m_temp_last_batch = torch.FloatTensor(last_batch_size, BNML.sum_cardinalities).zero_().to(
                para['device'])



            # target_temp = torch.FloatTensor(para['batch_size'], BNML.list_cardinalities[i]).zero_().to(para['device'])
            target_temp_last_batch = torch.FloatTensor(last_batch_size, BNML.list_cardinalities[i]).zero_().to(
                para['device'])
            if BNML.list_correspond_observed_variables[i] == -1: # 与隐变量无关的变量
                # data_s_temp = torch.FloatTensor(para['batch_size'], BNML.sum_cardinalities).zero_().to(para['device'])
                data_s_temp_last_batch = torch.FloatTensor(last_batch_size, BNML.sum_cardinalities).zero_().to(
                    para['device'])
            else:
                ov = BNML.list_correspond_observed_variables[i]  # 与隐变量有关的变量
                # print(ov)
                # data_s_temp = torch.FloatTensor(para['batch_size'] * BNML.list_cardinalities[ov],
                #                                 BNML.sum_cardinalities).zero_().to(para['device'])
                data_s_temp_last_batch = torch.FloatTensor(last_batch_size * BNML.list_cardinalities[ov],
                                                           BNML.sum_cardinalities).zero_().to(para['device'])
            # print(data_m_temp.shape)
            # print(data_m_temp_last_batch.shape)
            # print(target_temp.shape)
            # print(target_temp_last_batch.shape)
            # print(data_s_temp.shape)
            # print(data_s_temp_last_batch.shape)
            # exit(0)
            #######################################################################################
            for batch_idx in range(len(tarin_dl_tensor_list)):  # batch_idx: 0~n
                # print(batch_idx)
                # t1 = time.time()
                input, U_index, I_index, R = tarin_dl_tensor_list[batch_idx]
                # print('input')
                # print(input[0])
                # print(input.shape)
                # print(U_index)
                # print(I_index)
                # print(R)
                # exit(0)

                # t1 = time.time()
                if batch_idx == len(tarin_dl_tensor_list) - 1:
                    data_m = data_m_temp_last_batch.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
                    target = target_temp_last_batch.clone().detach()
                # else:
                #     data_m = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
                #     target = target_temp.clone().detach()
                # t2 = time.time()
                # print("time1=", "{:.4f}".format(t2 - t1))
                for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
                    if BNML.edges[j][i] == 1:  # 给父节点对应的行赋值
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = input[:,
                                                                                                           sum(BNML.list_cardinalities[
                                                                                                               :j]):sum(
                                                                                                               BNML.list_cardinalities[
                                                                                                               :j + 1])]
                    if i == j:  # 给当前节点输出赋值
                        target = input[:, sum(BNML.list_cardinalities[:i]):sum(BNML.list_cardinalities[:i + 1])]
                # print(data_m)
                # print(data_m.shape)
                # print(target)
                # print(target.shape)
                # exit(0)

                # t3 = time.time()
                # print("time2=", "{:.4f}".format(t3 - t2))
                ##################################################################################
                ##################################################################################
                w = None
                list_latent = []
                for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
                    if BNML.edges[j][i] == 1 or i == j:  # 父节点或当前节点j
                        if BNML.list_latent_variables[j] == 1:
                            list_latent.append(j)
                # print(list_latent)
                # exit(0)
                # t_em_e1 = time.time()
                if list_latent != []:
                    data_m, target, w = filling_Data(BNML, i, list_latent, data_m, target, para,
                                                     [U_index, I_index, R], net, df, w_temp)
                    if w_temp == None:
                        w_temp = w
                # print(w)
                # t_em_e2 = time.time()
                # print("t_em_e=", "{:.4f}".format(t_em_e2 - t_em_e1))
                # t4 = time.time()
                # print("time3=", "{:.4f}".format(t4 - t3))
                # data_s = torch.FloatTensor(len(data_m), BNML.sum_cardinalities).zero_().to(para['device'])
                if batch_idx == len(tarin_dl_tensor_list) - 1:
                    data_s = data_s_temp_last_batch.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
                # else:
                #     data_s = data_s_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
                # t5 = time.time()
                # print("time4=", "{:.4f}".format(t5 - t4))
                # exit(0)
                ##################################################################################
                ##################################################################################
                prior = None
                ##################################################################################
                ##################################################################################
                # print(data_m)
                # print(data_m.shape)
                # print(data_s)
                # print(data_s.shape)
                # print(target)
                # print(target.shape)
                # print(w)
                # print(w.shape)
                # exit(0)
            ##################################################################################
            ##################################################################################

            best_iter = -1
            best_loss = np.inf
            last_iter_loss = np.inf
            convergence_iter_num = 0
            for epoch in range(1, para['BN_parameter_learning_max_iter_num'] + 1):
                # print(epoch)
                train_loss = 0

                BNML.list_NPN[i].train()
                BN_loss = BNML.list_NPN[i].loss(data_m, data_s, target, w, prior)
                # t7 = time.time()
                # print("time6=", "{:.4f}".format(t7 - t6))
                optimizer.zero_grad()  # 清空过往梯度
                BN_loss.backward()  # 反向传播，计算当前梯度
                # t8 = time.time()
                # print("time7=", "{:.4f}".format(t8 - t7))
                optimizer.step()  # 根据梯度更新网络参数
                # t9 = time.time()
                # print("time8=", "{:.4f}".format(t9 - t8))
                train_loss += BN_loss.item()
                # if epoch % para['print_interval'] == 0:
                # if 1:
                #     print('Node: {}, Train Epoch: {}, BN loss: {:.4f}, EM Batch size: {}\n'.format(i + 1, epoch, BN_loss, len(data_m)))

                # early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    # best_iter = epoch
                    convergence_iter_num = 0
                else:
                    convergence_iter_num += 1
                # if convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num']:
                #     print('Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
                #     BNML.list_BIC_loglikelihood[i] = train_loss * -1

                # print(train_loss)
                # print(last_iter_loss)
                # print(last_iter_loss - train_loss)
                # print(1.0 / len(data_m))
                # if epoch == para['BN_parameter_learning_max_iter_num']:
                # if epoch > 50 and (convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num']):
                if epoch > 50 and (abs(last_iter_loss - train_loss) < 0.001 or (convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num'])):
                # if epoch > 50 and (abs(last_iter_loss - train_loss) < 0.0001 or (convergence_iter_num == para['BN_max_convergence_iter_num'] or epoch == para['BN_parameter_learning_max_iter_num'])):
                    print('Node: {}, Epoch: {}, best_loss: {:.2f}, EM Batch size: {}, best_iter: {}, convergence_iter_num: {}'.format(i + 1, epoch, best_loss, len(data_m), best_iter, convergence_iter_num))
                    BNML.list_BIC_loglikelihood[i] = train_loss * -1
                    break
                else:
                    last_iter_loss = train_loss
                    # best_iter = epoch

            # exit(0)

        return BNML, None