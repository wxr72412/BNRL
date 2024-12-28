import torch
from func.find_cpd import find_cpt
import numpy as np
import time

def get_theta_L(BNML, q, L, list_pa_L, list_ch_L, temp_data_m, temp_target, para, input_other, net, df):
    # print(q)
    # print(L)
    # print(list_pa_L)
    # print(list_ch_L)
    # exit(0)

    # print(temp_data_m)
    # print(temp_data_s)
    # print(temp_target)


    P_L_Pa = torch.FloatTensor(len(temp_target), 1).zero_().to(para['device'])
    P_Ch_L = torch.FloatTensor(len(temp_target), 1).zero_().to(para['device'])
    # print(P_L_Pa)
    # print(P_Ch_L)


    # temp_data_m_L = temp_data_m.clone().detach()  # 避免修改temp_data_m的时候将data_m修改
    # temp_data_m_L[:, sum(BNML.list_cardinalities[:L_index]):sum(BNML.list_cardinalities[:L_index + 1])] = temp_target


    if list_pa_L != []:
        pa_L = list_pa_L[0]
        net_pa_L = str(para['list_node_index_VQVAE'][pa_L])
        # print('pa_L: ' + str(pa_L))
        # print('net_pa_L: ' + net_pa_L)

        net_L = str(para['list_node_index_VQVAE'][L])
        # print('L: ' + str(L))
        # print('net_L: ' + net_L)

        ch_L = list_ch_L[0]
        net_ch_L = str(para['list_node_index_VQVAE'][ch_L])
        # print('ch_L: ' + str(ch_L))
        # print('net_ch_L: ' + net_ch_L)

        # print(df)
        pa_L_values = df[net_pa_L].values # <class 'numpy.ndarray'>
        L_values = q.cpu().numpy().astype('int64')
        ch_L_values = df[net_ch_L].values # <class 'numpy.ndarray'>
        # print(type(df_pa_L))
        # print(pa_L_values)
        # print(L_values)
        # print(ch_L_values)
        # exit(0)
        t1 = time.time()
        for a, L_value, pa_L_value in zip(range(len(pa_L_values)), L_values, pa_L_values):
            # print(net_pa_L)
            # print(pa_L_value)
            # print(net_L)
            # print(L_values)
            P_L_Pa[a] = find_cpt(net, list(net_pa_L), [pa_L_value], net_L, L_value)

        t2 = time.time()
        print("time1-1-1=", "{:.4f}".format(t2 - t1))
        for a, L_value, ch_L_value in zip(range(len(pa_L_values)), L_values, ch_L_values):
            # print(net_L)
            # print(L_value)
            # print(net_ch_L)
            # print(ch_L_value)
            t21 = time.time()
            P_Ch_L[a] = find_cpt(net, list(net_L), [L_value], net_ch_L, ch_L_value)
            t22 = time.time()
            print("time1-1-1-1=", "{:.4f}".format(t22 - t21))
        t3 = time.time()
        print("time1-1-2=", "{:.4f}".format(t3 - t2))

        P_LX = P_L_Pa.mul(P_Ch_L)

        # print(P_L_Pa)
        # print(P_Ch_L)
        # print(P_LX)
        # exit(0)

    else: #  list_pa_L == []
        pa_L = []
        net_pa_L = []
        # print('pa_L: ' + str(pa_L))
        # print('net_pa_L: ' + net_pa_L)

        net_L = str(para['list_node_index_VQVAE'][L])
        # print('L: ' + str(L))
        # print('net_L: ' + net_L)

        ch_L = list_ch_L[0]
        net_ch_L = str(para['list_node_index_VQVAE'][ch_L])
        # print('ch_L: ' + str(ch_L))
        # print('net_ch_L: ' + net_ch_L)

        # print(df)
        pa_L_values = []
        L_values = q.cpu().numpy().astype('int64')
        ch_L_values = df[net_ch_L].values  # <class 'numpy.ndarray'>
        # print(type(df_pa_L))
        # print(pa_L_values)
        # print(L_values)
        # print(ch_L_values)
        # exit(0)

        # print(L_values)
        # for a, L_value in zip(range(len(L_values)), L_values):
        #     # print(net_pa_L)
        #     # print(pa_L_value)
        #     # print(net_L)
        #     # print(L_values)
        #     P_L_Pa[a] = find_cpt(net, [], [], net_L, L_value)
        # t1 = time.time()
        P_L_Pa += find_cpt(net, [], [], net_L, L_values[0]) # 所有隐变量的值都相同，且没有父节点，所以cpt的值也相同
        # print(P_L_Pa)
        # exit(0)

        # print('TTT 1')
        # print(net_L)
        # print(L_values)
        # print(net_ch_L)
        # print(ch_L_values)
        # t2 = time.time()
        # print("time1-1-1=", "{:.4f}".format(t2 - t1))
        for a, L_value, ch_L_value in zip(range(len(L_values)), L_values, ch_L_values):
            # print(net_L)
            # print(L_value)
            # print(net_ch_L)
            # print(ch_L_value)
            # t21 = time.time()
            P_Ch_L[a] = find_cpt(net, list(net_L), [L_value], net_ch_L, ch_L_value)
            # t22 = time.time()
            # print("time1-1-1-1=", "{:.4f}".format(t22 - t21))
            # print(P_Ch_L[a])
            # exit(0)
        t3 = time.time()
        # print("time1-1-2=", "{:.4f}".format(t3 - t2))
        # print('P_Ch_L')
        # print(P_Ch_L)
        # tensor([[0.4980],
        #         [0.1450],
        #         [0.3570],
        #         ...,
        #         [0.4980],
        #         [0.3570],
        #         [0.3570]], device='cuda:0')
        # P_Ch_L
        # tensor([[0.4900],
        #         [0.4900],
        #         [0.3010],
        #         [0.4900],
        #         [0.4900],
        #         [0.4900],
        # exit(0)

        P_LX = P_L_Pa.mul(P_Ch_L)
        # print(P_LX)
        # exit(0)


    return P_LX
