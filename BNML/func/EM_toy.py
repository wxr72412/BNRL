import torch
import time
from func.toy_origin_theta import get_theta_L
import numpy as np

def filling_Data(BNML, i, list_latent, data_m, target, para, input_other, net, df, w_temp):
    # print(i) # 当前节点
    # print(list_latent) # 与当前节点相关的隐变量
    # print(data_m)
    # print(target)
    # exit(0)

    list_data_m_EM = []
    list_target_EM = []
    list_P_LX = []


    L_index = list_latent[0]
    # ov = BNML.list_correspond_observed_variables[L_index]  # 与隐变量节点对应的变量
    # print('lv: ' + str(lv))
    # print('ov: ' + str(ov))
    # exit(0)

    # temp_data_s = torch.FloatTensor(temp_data_m.shape[0], temp_data_m.shape[1]).zero_().to(para['device'])
    # print(temp_data_s[0])
    # print(temp_data_s.shape)


    if i == L_index:  # 隐变量是当前节点，则填充target
        list_pa_L = BNML.list_parent_nodes(i)
        list_ch_L = BNML.list_child_nodes(i)
    else:  # 隐变量是当前节点的父节点，则填充data_m
        list_pa_L = BNML.list_parent_nodes(L_index)
        list_ch_L = [i]
    # print(list_pa_L)
    # print(L_index)
    # print(list_ch_L)
    # exit(0)

    for index in range(BNML.list_cardinalities[L_index]):  # 将对应显变量的取值逐一填充到隐变量取值中
        temp_data_m = data_m.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
        temp_target = target.clone().detach()
        q = torch.ones(temp_target.shape[0]).to(para['device']) * index
        # print(q) # tensor([0., 0.], device='cuda:0')
        if i == L_index:  # 隐变量是当前节点，则填充target
            temp_target = torch.FloatTensor(temp_target.shape[0], BNML.list_cardinalities[L_index]).zero_().to(
                para['device']).scatter_(1, q.to(int).unsqueeze(1), 1)
            # print(temp_target)
            # tensor([[1., 0., 0., 0.],
            #         [1., 0., 0., 0.]], device='cuda:0')
            # exit(0)
        else: # 隐变量是父节点，则填充data_m
            temp_data_m[:, sum(BNML.list_cardinalities[:L_index]):sum(BNML.list_cardinalities[:L_index + 1])] = \
                torch.FloatTensor(temp_data_m.shape[0], BNML.list_cardinalities[L_index]).zero_().to(para['device']).scatter_(1, q.to(int).unsqueeze(1), 1)  # 将tensor([1., 1.])变为one-hot形式

        if w_temp == None:
            # t1 = time.time()
            P_LX = get_theta_L(BNML, q, L_index, list_pa_L, list_ch_L, temp_data_m, temp_target, para, input_other, net, df)
            # t2 = time.time()
            # print("time1-1=", "{:.4f}".format(t2 - t1))
            list_P_LX.append(P_LX)

        list_data_m_EM.append(temp_data_m)
        list_target_EM.append(temp_target)
    # print(list_P_LX)
    # print(list_data_m_EM)
    # print(list_target_EM)

    if w_temp == None:
        P_X = np.sum(list_P_LX, axis = 0)
        # print(P_X)
        # exit(0)

        list_P_L_X = [P_LX.div(P_X) for P_LX in list_P_LX]
        # print(list_P_L_X)
        # exit(0)

        w = torch.cat(list_P_L_X, dim=0) # 将隐变量不同取值合并在一起
        # print(w)
        # print(w.shape)  # torch.Size([8, 1])
        # exit(0)
    else:
        w = w_temp


    data_m_EM = torch.cat(list_data_m_EM, dim=0) # 将隐变量不同取值合并在一起
    # print(data_m_EM)
    # print(data_m_EM.shape)
    # exit(0)

    target_EM = torch.cat(list_target_EM, dim=0)
    # print(target_EM)
    # print(target_EM.shape)
    # exit(0)

    return data_m_EM, target_EM, w

