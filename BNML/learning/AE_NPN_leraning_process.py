import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from func.metrics import get_scores
import time
from data.loaddata import load_tarin_dl
from func.EM import filling_Data

loss_func_MSE = nn.MSELoss()
np.set_printoptions(threshold=np.inf)

# from torchviz import make_dot, make_dot_from_trace


def AE_NPN_learning(VQVAE, VQVAE_type, optimizer, epoch, para, features,
                    BNML = None, i = None, data_m_origin = None, data_s_origin = None, target_origin = None, input_other = None, test_dl = None):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    # t0 = time.time()
    train_loss = 0
    BN_loss = 0
    optimizer.zero_grad()
    if VQVAE_type == True:
        VQVAE.train()
        Z, z_e, z_q, emb, X_pred = VQVAE(features)
    ####################################################################################################################
    ####################################################################################################################
    if i != None:
        data_m = data_m_origin.clone().detach()
        data_s = data_s_origin.clone().detach()
        target = target_origin.clone().detach()
        # print(data_m_origin)
        # print(data_s_origin)
        # print(target_origin)
        # print(data_m)
        # print(data_s)
        # print(target)
        # print(input_other)
        # exit(0)

        if para['data_file']  == 'ml-1m':
            BNML.list_NPN[i].train()
            for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
                if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
                    if BNML.list_node_label[j] == 'eDUA':
                        index = 0
                        for k in range(BNML.num_nodes):
                            if BNML.list_node_label[k] == 'eDUA':
                                if k == j:
                                    break
                                if k != j:
                                    index += 1
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
                            torch.FloatTensor(VQVAE.indexU.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexU[:, index].unsqueeze(1), 1)[input_other[0]]
                    elif BNML.list_node_label[j] == 'eDIA':
                        index = 0
                        for k in range(BNML.num_nodes):
                            if BNML.list_node_label[k] == 'eDIA':
                                if k == j:
                                    break
                                if k != j:
                                    index += 1
                        data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
                            torch.FloatTensor(VQVAE.indexI.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexI[:, index].unsqueeze(1), 1)[input_other[1]]
                    # if BNML.list_node_label[j] == 'eDUA':
                    #     index = 0
                    #     for k in range(BNML.num_nodes):
                    #         if BNML.list_node_label[k] == 'eDUA':
                    #             if k == j:
                    #                 break
                    #             if k != j:
                    #                 index += 1
                    #     data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[0][input_other[0]][...,index]
                    #     # print(data_m[0]) # torch.Size([8])
                    #     # exit(0)
                    # elif BNML.list_node_label[j] == 'eDIA':
                    #     index = 0
                    #     for k in range(BNML.num_nodes):
                    #         if BNML.list_node_label[k] == 'eDIA':
                    #             if k == j:
                    #                 break
                    #             if k != j:
                    #                 index += 1
                    #     data_m[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[1][input_other[1]][...,index]
                if i == j: # 给当前节点输出赋值
                    if BNML.list_node_label[j] == 'eDUA':
                        index = 0
                        for k in range(BNML.num_nodes):
                            if BNML.list_node_label[k] == 'eDUA':
                                if k == j:
                                    break
                                if k != j:
                                    index += 1
                        target = torch.FloatTensor(VQVAE.indexU.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexU[:, index].unsqueeze(1), 1)[input_other[0]]
                    elif BNML.list_node_label[j] == 'eDIA':
                        index = 0
                        for k in range(BNML.num_nodes):
                            if BNML.list_node_label[k] == 'eDIA':
                                if k == j:
                                    break
                                if k != j:
                                    index += 1
                        target = torch.FloatTensor(VQVAE.indexI.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexI[:, index].unsqueeze(1), 1)[input_other[1]]
                    # if BNML.list_node_label[j] == 'eDUA':
                    #     index = 0
                    #     for k in range(BNML.num_nodes):
                    #         if BNML.list_node_label[k] == 'eDUA':
                    #             if k == j:
                    #                 break
                    #             if k != j:
                    #                 index += 1
                    #     target = z_q[0][input_other[0]][...,index]
                    #     # print(data_m[0]) # torch.Size([8])
                    #     # exit(0)
                    # elif BNML.list_node_label[j] == 'eDIA':
                    #     index = 0
                    #     for k in range(BNML.num_nodes):
                    #         if BNML.list_node_label[k] == 'eDIA':
                    #             if k == j:
                    #                 break
                    #             if k != j:
                    #                 index += 1
                    #     target = z_q[1][input_other[1]][...,index]
            # output = BNML.list_NPN[i]((data_m, data_s))
            # print(data_m[0])
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
            # print(input_other[0])
            # print(VQVAE.indexI)
            # print(input_other[1])
            # print(input_other[2])
            # print(VQVAE.indexI[8])
            # print(VQVAE.indexI[9])

            # print(next(VQVAE.parameters()).device)
            # print(data_m.device)
            # exit(0)
    ##################################################################################
    ##################################################################################
    w = None
    list_latent = []
    for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
        if BNML.edges[j][i] == 1 or i == j: # 父节点或子节点j
            if BNML.list_latent_variables[j] == 1:
                list_latent.append(j)
    if list_latent != []:
        data_m, data_s, target, w = filling_Data(BNML, i, list_latent, data_m, target, para, input_other, VQVAE)
    # print(data_m)
    # print(data_m[0])
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
    ########################################### 用于验证损失函数是佛正确 ###################################################
    ########################################### 用于验证损失函数是佛正确 ###################################################
    # data_m = torch.FloatTensor(6, BNML.sum_cardinalities).zero_().to(para['device'])
    # data_m[0][0] = 1
    # data_m[1][0] = 1
    # data_m[2][0] = 1
    # data_m[3][1] = 1
    # data_m[4][1] = 1
    # data_m[5][1] = 1
    # data_s = torch.FloatTensor(data_m.shape[0], BNML.sum_cardinalities).zero_().to(para['device'])
    # target = torch.FloatTensor(data_m.shape[0], BNML.list_cardinalities[i]).zero_().to(para['device'])
    # target[0][0] = 1
    # target[1][0] = 1
    # target[2][1] = 1
    # target[3][0] = 1
    # target[4][1] = 1
    # target[5][1] = 1
    # w = torch.FloatTensor(data_m.shape[0], 1).zero_().to(para['device'])
    # w[0] = 0.7
    # w[1] = 0.7
    # w[2] = 0.3
    # w[3] = 0.3
    # w[4] = 0.7
    # w[5] = 0.7

    # data_m = torch.FloatTensor(3, BNML.sum_cardinalities).zero_().to(para['device'])
    # data_m[0][1] = 1
    # data_m[1][7] = 1
    # data_m[2][1] = 1
    # data_s = torch.FloatTensor(data_m.shape[0], BNML.sum_cardinalities).zero_().to(para['device'])
    # target = torch.FloatTensor(data_m.shape[0], BNML.list_cardinalities[i]).zero_().to(para['device'])
    # target[0][4] = 1
    # target[1][2] = 1
    # target[2][4] = 1
    # w = torch.FloatTensor(data_m.shape[0], 1).zero_().to(para['device'])
    # w[0] = 0.2
    # w[1] = 0.2
    # w[2] = 0.2

    #
    # prior = torch.FloatTensor(1, 5).zero_().to(para['device'])
    # prior[0][0] = 0.5
    # prior[0][1] = 0.5

    # print(data_m)
    # print(data_m[0])
    # print(data_m[1])
    # print(data_m[2])
    # print(data_s)
    # print(target)
    # print(w)
    # print(prior)
    # print(data_m.shape)
    # print(data_s.shape)
    # print(target.shape)
    # print(w.shape)
    # print(prior.shape)
    # exit(0)
    ##################################################################################
    ##################################################################################
    # t3 = time.time()
    # print("time2-3=", "{:.4f}".format(t3 - t2))
    BN_loss += BNML.list_NPN[i].loss(data_m, data_s, target, w, prior)
    # print('BN_loss')
    # print(BN_loss.item())
    # exit(0)
    train_loss += BN_loss

    # if VQVAE_type == True:
    #     VQVAE_loss = VQVAE.loss(X_pred, features, z_e, emb)
    #     train_loss += VQVAE_loss
    train_loss.backward()
    # print(BNML.list_NPN[i].layers[0].W_m.grad)
    # print(BNML.list_NPN[i].layers[0].M_s.grad)
    # print(BNML.list_NPN[i].layers[0].b_m.grad)
    # print(BNML.list_NPN[i].layers[0].p_s.grad)
    # print('model.base_encodeU')
    # print(VQVAE.base_encodeU[0].weight.grad)
    # print('model.embd')
    # print(VQVAE.embU.weight.grad)
    # print('model.decoderU')
    # print(VQVAE.decoderU[0].weight.grad)
    # exit(0)
    optimizer.step()
    ####################################################################################################################
    ####################################################################################################################
    # t6 = time.time()
    # print("time5-6=", "{:.4f}".format(t6 - t5))

    # if 1:
    # print(epoch)
    if epoch % para['print_interval'] == 0:
        print('Node: {}, Train Epoch: {}, Train loss: {:.4f}, Batch size: {}, EM Batch size: {}\n'.format(i+1, epoch, train_loss, len(data_m_origin), len(data_m)))

        if para['data_file'] == 'ml-1m':
        #     print("BN_loss:", '%04f' % (BN_loss))
        #     print("train_loss:", '%04f' % (train_loss))
        #     # torch.set_printoptions(precision=10)
            output = BNML.list_NPN[i]((data_m, data_s))
            print(output[0])
            # BNML.output_CPD_node_i(i, para)
        #     # out = output[0].reshape(-1)
        #     out = (output[0].argmax(dim=1)+1).reshape(-1)
        #
        #     # output = RSNN([z_q[0][input_other[0]].view(-1, para['hidden_dim_VQVAE_U']), z_q[1][input_other[1]].view(-1, para['hidden_dim_VQVAE_I'])])
        #     # output = RSNN([z_q[0][input_other[0]].view(-1, para['hidden_dim_VQVAE_U']), z_q[1][input_other[1]].view(-1, para['hidden_dim_VQVAE_I'])])
        #     # output = RSNN([data_m[:, 0:para['dim_U_feature']], data_m[:, para['dim_U_feature']:para['dim_U_feature']+para['dim_I_feature']]])
        #     # out = output[0].reshape(-1)
        #
        #     print(output[0])
        #     print(out.shape)
        #     print(input_other[2].shape)
        #     print(out)
        #     print(input_other[2])
        #     # exit(0)
        #     # loss = loss_func_MSE(output[0].reshape(-1), input_other[2].reshape(-1))
        #     # print(out - input_other[2].float().reshape(-1))
        #     mse = ((out - input_other[2].float().reshape(-1)) ** 2.).mean().item()
        #     mae = (torch.abs(out - input_other[2].float().reshape(-1))).mean().item()
        #     rmse = np.sqrt(((out - input_other[2].float().reshape(-1)) ** 2.).mean().item())
        #     print("train_MSE: ", "{:.6f}".format(mse))
        #     print("train_MAE:", "{:.6f}".format(mae))
        #     print("train_RMSE:", "{:.6f}".format(rmse))
        #     # print(input_other[0])
        #     for batch_idx, data in enumerate(test_dl): # batch_idx: 0~n
        #         input_test, U_index_test, I_index_test, R_test = load_tarin_dl(data, para, BNML, features)
        #         data_m_test = torch.FloatTensor(len(R_test), BNML.sum_cardinalities).zero_().to(para['device'])
        #         data_s_test = torch.FloatTensor(len(R_test), BNML.sum_cardinalities).zero_().to(para['device'])
        #         for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
        #             if BNML.edges[j][i] == 1: # 给父节点对应的行赋值
        #                 if BNML.list_node_label[j] == 'eDUA':
        #                     index = 0
        #                     for k in range(BNML.num_nodes):
        #                         if BNML.list_node_label[k] == 'eDUA':
        #                             if k == j:
        #                                 break
        #                             if k != j:
        #                                 index += 1
        #                     data_m_test[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
        #                         torch.FloatTensor(VQVAE.indexU.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexU[:, index].unsqueeze(1), 1)[U_index_test]
        #                 elif BNML.list_node_label[j] == 'eDIA':
        #                     index = 0
        #                     for k in range(BNML.num_nodes):
        #                         if BNML.list_node_label[k] == 'eDIA':
        #                             if k == j:
        #                                 break
        #                             if k != j:
        #                                 index += 1
        #                     data_m_test[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = \
        #                         torch.FloatTensor(VQVAE.indexI.shape[0], BNML.list_cardinalities[j]).zero_().to(para['device']).scatter_(1, VQVAE.indexI[:, index].unsqueeze(1), 1)[I_index_test]
        #                 # if BNML.list_node_label[j] == 'eDUA':
        #                 #     index = 0
        #                 #     for k in range(BNML.num_nodes):
        #                 #         if BNML.list_node_label[k] == 'eDUA':
        #                 #             if k == j:
        #                 #                 break
        #                 #             if k != j:
        #                 #                 index += 1
        #                 #     data_m_test[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[0][U_index_test][...,index]
        #                 #     # print(data_m[0]) # torch.Size([8])
        #                 #     # exit(0)
        #                 # elif BNML.list_node_label[j] == 'eDIA':
        #                 #     index = 0
        #                 #     for k in range(BNML.num_nodes):
        #                 #         if BNML.list_node_label[k] == 'eDIA':
        #                 #             if k == j:
        #                 #                 break
        #                 #             if k != j:
        #                 #                 index += 1
        #                 #     data_m_test[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = z_q[1][I_index_test][...,index]
        #         # print(data_m_test)
        #         # print(VQVAE.indexU)
        #         # print(U_index_test)
        #         # print(VQVAE.indexI)
        #         # print(I_index_test)
        #         # print(VQVAE.indexI[7])
        #         # print(VQVAE.indexI[3])
        #         # exit(0)
        #         output_test = BNML.list_NPN[i]((data_m_test, data_s_test))
        #         # out_test = output_test[0].reshape(-1)
        #         out_test = (output_test[0].argmax(dim=1)+1).reshape(-1)
        #
        #         # output_test = RSNN([z_q[0][U_index_test].view(-1, para['hidden_dim_VQVAE_U']), z_q[1][I_index_test].view(-1, para['hidden_dim_VQVAE_I'])])
        #         # output_test = RSNN([data_m_test[:, 0:para['dim_U_feature']], data_m_test[:, para['dim_U_feature']:para['dim_U_feature']+para['dim_I_feature']]])
        #         # out_test = output_test[0].reshape(-1)
        #
        #         print(output_test[0])
        #         print(out_test.shape)
        #         print(R_test.shape)
        #         print(out_test)
        #         print(R_test)
        #         # exit(0)
        #
        #         # print(out_test - R_test.float().reshape(-1))
        #         mse_test = ((out_test - R_test.float().reshape(-1)) ** 2.).mean().item()
        #         mae_test = (torch.abs(out_test - R_test.float().reshape(-1))).mean().item()
        #         rmse_test = np.sqrt(((out_test - R_test.float().reshape(-1)) ** 2.).mean().item())
        #         print("test_MSE: ", "{:.6f}".format(mse_test))
        #         print("test_MAE:", "{:.6f}".format(mae_test))
        #         print("test_RMSE:", "{:.6f}".format(rmse_test))
        #         # exit(0)




            if VQVAE_type == True:
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

            # for a,b in zip(A_pred[0].view(-1), adj[0].to_dense().view(-1)):
            #     print("{:.4f}".format(a.item()), "::", "{:.4f}".format(b.item()), "  ", end='')
            # print()

            # print(model.mean[0: para['userNum']].shape)
            # print(model.std[para['userNum']: para['userNum']+para['userNum']].shape)
            # exit(0)

    return train_loss.item(), BN_loss.item()