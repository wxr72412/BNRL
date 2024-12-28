import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from func.metrics import get_scores
import time
from data.loaddata import load_tarin_dl
from func.EM import filling_Data
import torch.optim as optim
from my_model.npn1 import GaussianNPN, vanillaNN
from func.find_cpd import find_cpt


loss_func_MSE = nn.MSELoss()
np.set_printoptions(threshold=np.inf)

# from torchviz import make_dot, make_dot_from_trace


def All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features, features_origin, VQVAE, VQVAE_type, net):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    # t0 = time.time()

    # list_P_L_Pa_NPN = [None for i in range(0, para['num_VQVAE'])]
    list_P_Ch_L_NPN = [None for i in range(0, para['num_VQVAE'])]
    if VQVAE_type == True:
        for n in range(para['num_VQVAE']):

            print('num_VQVAE: ' + str(n))
            params_list = []
            params_list.append(VQVAE.list_embed[n].parameters())
            params_list.append(VQVAE.list_base_encode[n].parameters())
            params_list.append(VQVAE.list_decoder[n].parameters())
            ##############################################################################################################################
            ##############################################################################################################################
            net_L = para['VQVAE_node_label'].index(str(n)) + 1
            # print('net_L: ' + str(net_L))
            num = para['VQVAE_node_label'].count(str(n))
            net_list_L = [str(l) for l in range(net_L, net_L + num)]
            # print('net_list_L: ' + str(net_list_L))

            # print(net['parents'][str(net_L)])
            if net['parents'][str(net_L)] != []:
                net_pa_L = net['parents'][str(net_L)][0]
            else:
                net_pa_L = []

            # print('net_pa_L: ' + str(net_pa_L))
            net_ch_L = net['children'][str(net_L)][0]
            # print('net_ch_L: ' + str(net_ch_L))
            # exit(0)

            # for num in range(len(input)):
            #
            #     P_Ch_L = find_cpt(net, pa, pa_values, V, V_value)
            #     print(P_Ch_L)
            #     data_m_P_Ch_L[num] = P_Ch_L
            # print(features_origin)
            # print(features_origin[:, net_L:net_L+num])
            # print(features_origin[:, int(net_Ch_L)])

            # print(net['cardinality'])
            if net_pa_L != []:
                net_pa_L_cardinality = net['cardinality'][str(net_pa_L)]
            else:
                net_pa_L_cardinality = 1
            net_Ch_L_cardinality = net['cardinality'][str(net_ch_L)]

            # net_pa_L_cardinalities = sum(para['node_cardinalities'][0])
            net_pa_L_cardinalities = len(para['node_cardinalities'][0]) * net_pa_L_cardinality


            # print(net_pa_L_cardinality)
            # print(net_Ch_L_cardinality)
            # print(net_pa_L_cardinalities)
            # exit(0)
            ##############################################################################################################################
            ##############################################################################################################################
            # list_P_L_Pa_NPN[n] = GaussianNPN(BNML.sum_cardinalities, net_pa_L_cardinalities, "C", para['layers_NPN'], para).to(para['device'])
            list_P_Ch_L_NPN[n] = GaussianNPN(BNML.sum_cardinalities, net_Ch_L_cardinality, "C", para['layers_NPN'], para).to(para['device'])
            # params_list.append(list_P_L_Pa_NPN[n].parameters())
            params_list.append(list_P_Ch_L_NPN[n].parameters())

            # print(len(params_list))
            # exit(0)

            optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr'], weight_decay=0) # lr

            for batch_idx in range(len(tarin_dl_tensor_list)):  # batch_idx: 0~n
                input, U_index, I_index, R = tarin_dl_tensor_list[batch_idx]
                # print(input)
                # print(U_index)
                # print(I_index)
                # print(R)
                # print(features)

                data_m_temp = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])
                data_s = torch.FloatTensor(len(input), BNML.sum_cardinalities).zero_().to(para['device'])

                # print(i)
                # exit(0)
                # for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
                #     if BNML.edges[j][first_L] == 1:  # 给父节点对应的行赋值
                #         if BNML.list_node_label[j] == 'V':
                #             data_m_P_L_Pa_temp[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = \
                #                 input[:,sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])]
                #     if BNML.edges[first_L][j] == 1:  # 给父节点对应的行赋值
                #         if BNML.list_node_label[j] == 'V':
                #             data_m_P_Ch_L_temp[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = \
                #                 input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])]

                # print(data_m_P_L_Pa_temp)
                # print(data_m_P_Ch_L_temp)
                # print(data_s)
                ##############################################################################################
                ##############################################################################################
                # target_P_L_Pa = torch.FloatTensor(len(input), net_pa_L_cardinalities).zero_().to(para['device'])
                target_P_Ch_L = torch.FloatTensor(len(input), net_Ch_L_cardinality).zero_().to(para['device'])
                # exit(0)

                for a, pa_values, V_value in zip(range(len(input)), features_origin[:, net_L:net_L+num], features_origin[:, int(net_ch_L)]):
                    # print(list(a))
                    # print(b)
                    P_Ch_L = find_cpt(net, net_list_L, list(pa_values), net_ch_L, None)
                    # print(P_Ch_L)
                    # print(torch.tensor(P_Ch_L))
                    # exit(0)
                    target_P_Ch_L[a] = torch.tensor(P_Ch_L)

                # print('target_P_Ch_L')
                # print(target_P_Ch_L)
                # for i in range(target_P_Ch_L.shape[0]):
                #     print(target_P_Ch_L[i])
                #
                #
                # exit(0)

                # print('features_origin')
                # print(features_origin)
                #
                # print('features_origin[:, net_L:net_L+num]')
                # print(features_origin[:, net_L:net_L+num])

                # if net_pa_L != []:
                #     for a, V_values, pa_value in zip(range(len(input)), features_origin[:, net_L:net_L + num],
                #                                      features_origin[:, int(net_pa_L)]):
                #         b = 0
                #         for L, V_value in zip(net_list_L, list(V_values)):
                #             for pa_value in range(net_pa_L_cardinality):
                #                 # print(net_pa_L)
                #                 # print(pa_value)
                #                 # print(net_pa_L)
                #                 # print([pa_value])
                #                 # print(L)
                #                 # print(V_value)
                #                 P_L_Pa = find_cpt(net, list(net_pa_L), [pa_value], L, V_value)
                #                 # print(P_L_Pa)
                #                 target_P_L_Pa[a][b] = P_L_Pa
                #                 b += 1
                #         # print(target_P_L_Pa)
                #         # exit(0)
                # else:
                #     for a, V_values in zip(range(len(input)), features_origin[:, net_L:net_L + num]):
                #         b = 0
                #         for L, V_value in zip(net_list_L, list(V_values)):
                #             P_L_Pa = find_cpt(net, [], [], L, V_value)
                #             # print(P_L_Pa)
                #             target_P_L_Pa[a][b] = P_L_Pa
                #             b += 1
                #         # print(target_P_L_Pa)
                #         # exit(0)
                # print(target_P_L_Pa)
                # print(target_P_L_Pa[0])
                # exit(0)
                ##############################################################################################
                ##############################################################################################
                first_L = para['list_node_label'].index(str(n))
                # print('first_L: '+ str(first_L) )

                best_iter = -1
                best_loss = np.inf
                convergence_iter_num = 0
                for epoch in range(1, para['AE_parameter_learning_max_iter_num'] + 1):
                    # print('epoch: ' + str(epoch))
                    train_loss = 0 # grad_fn=<AddBackward0>
                    VQVAE.train()
                    # list_P_L_Pa_NPN[n].train()
                    list_P_Ch_L_NPN[n].train()

                    Z, z_e, z_q, emb, X_pred = VQVAE(features, n)
                    # print(features)
                    # print(features)
                    # print(features)
                    # print(features)
                    # print(VQVAE.list_index[n].reshape(-1).tolist())
                    # print(Z)
                    # print(z_e)
                    # print(z_q)
                    # print(emb)
                    # print(X_pred)
                    # exit(0)
                    VQVAE_loss = VQVAE.loss(X_pred, features, z_e, emb, n)
                    train_loss += VQVAE_loss
                    # print(VQVAE_loss)
                    # print(train_loss)

                    # t1 = time.time()
                    data_m = data_m_temp.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
                    # print(data_m_P_L_Pa_temp)
                    # print(data_m_P_Ch_L_temp)
                    # t2 = time.time()
                    # print("time=", "{:.4f}".format(t2 - t1))

                    i = None
                    for j in range(BNML.num_nodes):  # 0 ~ num_nodes-1
                        if BNML.edges[first_L][j] == 1:
                            # print('j: ' + str(j))
                            if BNML.list_node_label[j] == 'V':
                                ch_L = j
                                # print(ch_L)
                    i = ch_L
                    # print('i: ' + str(i))
                    # exit(0)
                    for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                        if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                            # print('j: ' + str(j))
                            # print(BNML.list_node_label[j])
                            if BNML.list_node_label[j] != 'V':
                                index = 0
                                for k in range(BNML.num_nodes):
                                    if BNML.list_node_label[k] == str(n):
                                        if k == j:
                                            break
                                        if k != j:
                                            index += 1
                                if BNML.list_node_type[j] == "C":
                                    # print(data_m)
                                    # print(z_q[n])
                                    # print(U_index)
                                    # print(z_q[n][U_index][..., index])
                                    data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j + 1])] = z_q[n][U_index][..., index]
                                # print(emb)
                                # print(emb[V])
                                # print(U_index)
                                # print(z_q[V])
                                # print(z_q[V][U_index])
                                # print(index)
                                # print(z_q[V][U_index][..., index])
                                # print(VQVAE.list_index[V])
                                # print(data_m) # torch.Size([8])
                                # exit(0)
                    # print(data_m)
                    # print(data_m.shape)
                    # print(data_s)
                    # print(data_s.shape)
                    # print(prior)
                    # print(output)
                    # print(target_P_L_Pa)
                    # print(target_P_Ch_L)
                    #
                    # print(VQVAE.list_index[0])

                    # print(VQVAE.list_emb[0].weight)
                    # exit(0)



                    ##################################################################################
                    ##################################################################################
                    w = None
                    # list_latent = []
                    # for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                    #     if BNML.edges[j][i] == 1 or i == j: # 父节点或子节点j
                    #         if BNML.list_latent_variables[j] == 1:
                    #             list_latent.append(j)
                    # if list_latent != []:
                    #     data_m, data_s, target, w = filling_Data(BNML, i, list_latent, data_m, target, para, [U_index, I_index, R], VQVAE)

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
                    # exit(0)
                    ##################################################################################
                    ##################################################################################
                    prior = None
                    ##################################################################################
                    ##################################################################################
                    # BN_loss_P_L_Pa_NPN = list_P_L_Pa_NPN[n].loss(data_m, data_s, target_P_L_Pa, w, prior)
                    BN_loss_P_Ch_L_NPN = list_P_Ch_L_NPN[n].loss(data_m, data_s, target_P_Ch_L, w, prior)
                    # train_loss += (BN_loss_P_L_Pa_NPN) * para['BN_loss_ratio']
                    train_loss += (BN_loss_P_Ch_L_NPN) * para['BN_loss_ratio']
                    # train_loss += (BN_loss_P_L_Pa_NPN + BN_loss_P_Ch_L_NPN) * para['BN_loss_ratio']
                    # train_loss += BN_loss / (BN_loss / VQVAE_loss).detach()

                    # if epoch % para['print_interval'] == 0:
                    # # if 1:
                    #     # print('Node: {}, Train Epoch: {}, BN loss: {:.4f}, Batch size: {}, EM Batch size: {}\n'.format(i+1, epoch, BN_loss_P_L_Pa_NPN + BN_loss_P_Ch_L_NPN, len(input), len(data_s)))
                    #     output = list_P_L_Pa_NPN[n]((data_m, data_s))
                    #     print(output[0])
                    #     # output = list_P_Ch_L_NPN[n]((data_m, data_s))
                    #     # print(output[0])

                    # print(BNML.list_BIC_loglikelihood)
                    # print(train_loss)

                    optimizer.zero_grad()  # 清空过往梯度
                    # print(train_loss)
                    # print('model.base_encode')
                    # print(VQVAE.list_base_encode[0][0].weight.grad)
                    # print('model.embd')
                    # print(VQVAE.list_emb[0].weight.grad)
                    # print('model.decoder')
                    # print(VQVAE.list_decoder[0][0].weight.grad)
                    train_loss.backward()  # 反向传播，计算当前梯度
                    # print(train_loss)
                    # print('model.base_encode')
                    # print(VQVAE.list_base_encode[0][0].weight.grad)
                    # print('model.embd')
                    # print(VQVAE.list_emb[0].weight.grad)
                    # print('model.decoder')
                    # print(VQVAE.list_decoder[0][0].weight.grad)
                    optimizer.step()  # 根据梯度更新网络参数
                    # exit(0)


                    # if 1:
                    if epoch % para['print_interval'] == 0:
                        print('Epoch: {}, Train loss: {:.6f}'.format(epoch, train_loss))
                        if VQVAE != None:
                            print("epoch:", '%d' % (epoch), "VQVAE_loss:", '%6f' % (VQVAE_loss))
                            print("BCE_X=", "{:.6f}".format(F.binary_cross_entropy(X_pred[n].reshape(-1), features[n].reshape(-1), weight=None)),
                                  "MSE_X=", "{:.6f}".format(loss_func_MSE(X_pred[n].reshape(-1), features[n].reshape(-1))))

                # print(train_loss_all)
                # print(epoch)
                # early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_iter = epoch
                    convergence_iter_num = 0
                else:
                    convergence_iter_num += 1
                if convergence_iter_num == para['AE_parameter_learning_max_iter_num'] or epoch == para['AE_parameter_learning_max_iter_num']:
                    print('Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
        return BNML, VQVAE