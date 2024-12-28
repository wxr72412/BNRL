# from init_para import toy1_init_para as init
from init_para import toy2_init_para as init

# from init_para import Child1_init_para as init
from init_para import child_L2_init_para as init

import torch
import numpy as np
import time
import para_init
from learning.structure_leraning import structure_train_process
from learning.CPD_learning_process import CPD_train
from learning.AE_train_process import AE_train1
from my_model.BN import BN, load_BN
from data.loaddata import load_data, process_features, load_BN_trainset
from my_model import AEs
from data.loaddata import load_tarin_dl
from learning.toy_All_AE_NPN_leraning_process_C import All_AE_NPN_learning
import torch.nn as nn
import torch.nn.functional as F
import os
from BN.my_BIF import BIFReader as my_BIFReader
from BN.my_BIF import BIFWriter as my_BIFWriter
from pgmpy.readwrite.BIF import BIFReader, BIFWriter
from BN import my_bayesnet
from BN.get_missing_index import get_missing_index


loss_func_MSE = nn.MSELoss()

if __name__ == '__main__':
    para = init.para
    para = init.func(para, 'C')
    # print(para)
    # print(para['list_node_type'])
    # print(para['list_cardinalities'])
    # exit(0)
    if para['device'] == "gpu":
        para['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(para['list_edges'])
    # print(para)
    # exit(0)

    SEED = para_init.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    num_latent = para['num_latent_variables']

    path = os.path.dirname(os.path.dirname(__file__)) + "\\data\\" + para['data_file'] + '\\'
    # data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + para['file_name_train']
    data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\' + para['file_name_train']
    net_bif = path

    if os.path.exists(path + 'VQVAE'):
        pass
    else:
        os.mkdir(path + 'VQVAE')

    if os.path.exists(path + 'VQVAE\\' + str(para['num_sample'])):
        pass
    else:
        os.mkdir(path + 'VQVAE\\' + str(para['num_sample']))

    if os.path.exists(path + 'VQVAE\\' + str(para['num_sample']) + '\\' + str(num_latent)):
        pass
    else:
        os.mkdir(path + 'VQVAE\\' + str(para['num_sample']) + '\\' + str(num_latent))

    VQVAE_path = path + 'VQVAE\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'
    # fulldata_path = path + 'fulldata\\' + str(para['num_sample']) + '\\'
    fulldata_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'
    missingdata_path = path + 'missingdata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'
    # net_bif = data_path + '/data/bn/remove_edges_rename_child.bif'
    # dat_records = data_path + '\\data\\missingdata\\1000\\4\\child_1000.xlsx'
    # print(data_path)
    # print(num_latent)
    # print(missingdata_path)
    # exit(0)
    ################################################################################################################
    ################################################################################################################
    if para['Train'] == 'True':
        tl = time.time()
        print("------------creating BNML!-------------------")
        num_nodes = para['num_nodes'] # BNML变量个数
        init_edges = np.zeros((num_nodes, num_nodes), dtype = int, order='C') # 初始边的邻接矩阵
        constraint_may_edges = np.zeros((num_nodes, num_nodes), dtype = int, order='C') # 初始边的邻接矩阵
        for e in para['list_edges']:
            init_edges[e[0]-1][e[1]-1] = 1
        for e in para['list_may_edges']:
            constraint_may_edges[e[0]-1][e[1]-1] = 1
        # print(init_edges)
        # print(constraint_may_edges)
        BNML = BN(num_nodes, para['list_node_type'], para['list_node_label'], para['list_cardinalities'],
                  para['list_latent_variables'], para['list_correspond_observed_variables'], init_edges, constraint_may_edges, para)
        BNML.create_CPD(para['device'])

        # for i in range(BNML.num_nodes):  # 0 ~ num_nodes-1
        #     print(next(BNML.list_NPN[i].parameters()).is_cuda)  # True

        print('BNML.num_nodes: ' + str(BNML.num_nodes))
        print('BNML.edges:')
        print(BNML.edges)
        # print('BNML.constraint_may_edges:')
        # print(BNML.constraint_may_edges)
        ################################################################################################################
        ################################################################################################################
        features_L = None
        VQVAE = None
        print("---------------加载特征数据！---------------------")
        features_origin = load_data(para, 'feature', data_file_path)
        # print(features_origin)
        print(features_origin.shape)
        # print(type(features_origin))
        print(features_origin[0])
        print(features_origin[1])
        print(features_origin[2])

        features_L = process_features(para, BNML, features_origin)
        # print(features_L)
        print(features_L[0][0])
        print(features_L[0][1])
        print(features_L[0][2])
        print(features_L[0].shape)  # torch.Size([12, 4])
        # print(features_origin[0])
        # exit(0)

        # features = np.zeros((para['sum_node_cardinalities'][0], para['num_var'] + 1), dtype=int)
        # list_temp = np.array([i for i in range(0, para['sum_node_cardinalities'][0])])
        # features[:, 0] = list_temp
        # list_temp = list_temp
        # for L in range(para_init.num_latent, 0, -1):
        #     print(L)
        #     car_comb = para['node_cardinalities_combinations'][0][L-1]
        #     # print(list_temp)
        #     # print(car_comb)
        #     if L != 1:
        #         a = list_temp / car_comb
        #         list_temp = list_temp % car_comb
        #         features[:, L] = a
        #         # print(a)
        #         # print(list_temp)
        #     else:
        #         features[:, L] = list_temp
        # print(features)
        # print(features.shape)
        # print(features[:, 0])
        # # exit(0)
        #
        #
        # features_L = process_features(para, BNML, features)
        # print(features_L)
        # print(features_L[0][0])
        # print(features_L[0][1])
        # print(features_L[0][2])
        # print(features_L[0].shape)  # torch.Size([12, 4])
        # print(features_origin[0])



        if "0" in BNML.list_node_label:
            print("------------learning VQVAE!-------------------")
            VQVAE = getattr(AEs, 'VQVAE')(para, features_L).to(para['device'])
            # print(VQVAE)
            # print(next(VQVAE.parameters()).is_cuda)
            # print(next(VQVAE.list_emb[0].parameters()).is_cuda)  # True
            # print(next(VQVAE.list_base_encode[0].parameters()).is_cuda)  # True
            # print(next(VQVAE.list_decoder[0].parameters()).is_cuda)  # True
            # exit(0)
                # print(VQVAE)
                # train_loss, Z = AE_train1(VQVAE, True,  para, features)
                # AEs.save_AE(para, VQVAE, "VQVAE")
        # for a, b in zip(X_pred[0][0].view(-1), features[0][0].view(-1)):
        #     print("{:.4f}".format(a.item()), "::", "{:.4f}".format(b.item()), "  ", end='')
        # print()
        # for a, b in zip(X_pred[1][0].view(-1), features[1][0].view(-1)):
        #     print("{:.4f}".format(a.item()), "::", "{:.4f}".format(b.item()), "  ", end='')
        # print()
        # exit(0)
        ###############################################################################################################
        ###############################################################################################################
        print("---------------加载BN训练数据！---------------------")
        train_dl, test_dl, trainLen, testLen = load_BN_trainset(para, test_batch_size = None, shuffle = False, data_file_path = data_file_path)
        print('trainLen:' + str(trainLen))
        print('testLen:' + str(testLen))
        para['train_size'] = trainLen

        tarin_dl_tensor_list = []
        for batch_idx, data in enumerate(train_dl): # batch_idx: 0~n
            input, index, I_index, R = load_tarin_dl(data, para, BNML, features_L)
            tarin_dl_tensor_list.append([input, index, I_index, R])
        # print(tarin_dl_tensor_list)
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        print("0: Reading Network . . . ")
        print(fulldata_path)
        reader = my_BIFReader(fulldata_path + para['data_file'] + '.bif')
        my_net = reader.my_model()
        # print(my_net['learn_cpds_df'])

        print("1: Getting data from records . . . ")
        df = my_bayesnet.read_excel_data(missingdata_path + para['data_file'] + '.xlsx')
        print(df)
        # Initialise parameters

        print("3: Getting missing data indexes . . . ")
        mis_index, mis_var = get_missing_index(df)
        print(mis_index)
        print(mis_var)
        # exit(0)

        # Initialise parameters
        print("2: Initialising parameters . . . ")
        net = my_bayesnet.my_init_param(df, my_net, mis_var)
        # print(net['V'])
        # for v in net['V']:
        # print(net['cpds'][v])
        # print(net['cpds_df'][v])
        #     print(net['learn_cpds_df'][v])
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        print("---------------BNML learning---------------------")
        t_start = time.time()
        BNML, VQVAE = All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features_L, features_origin, VQVAE, True, my_net)
        t_stop = time.time()
        print("VQVAE Time: " + str(round((t_stop - t_start), 4)))
        # print(VQVAE)
        AEs.save_AE(para, VQVAE, "VQVAE-files", VQVAE_path)
        # exit(0)

        for i in range(0, BNML.num_nodes):
            BNML.list_BIC_penalty[i] = BNML.independent_parameters_node_i(i) / 2 * np.log(para['train_size'])
            BNML.list_BIC[i] = BNML.list_BIC_loglikelihood[i] - BNML.list_BIC_penalty[i]

        if para['Save_model'] == True:
            BNML.save_BN_files(para, 'C', VQVAE_path)

        ################################################################################################################
        ################################################################################################################
        print("list_BIC_loglikelihood:")
        print(BNML.list_BIC_loglikelihood)
        print("list_BIC_penalty:")
        print(BNML.list_BIC_penalty)
        print("list_BIC:")
        print(BNML.list_BIC)
        print()
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        print("---------------BNML - VQVAE---------------------")
        df = load_data(para, 'df', data_file_path).astype('str')
        reader = BIFReader(fulldata_path + para['data_file'] + '.bif', n_jobs=1)
        net = reader.get_model()
        print(df)
        # print(df.dtypes)
        # exit(0)

        Z, z_e, z_q, emb, X_pred = VQVAE(features_L)
        # print(VQVAE.list_index)
        # exit(0)
        for n in range(para['num_VQVAE']):
            net_L = para['VQVAE_node_label'].index(str(n)) + 1
            num = para['VQVAE_node_label'].count(str(n))
            net_list_L = [str(l) for l in range(net_L, net_L + num)]
            if my_net['parents'][str(net_L)] != []:
                net_pa_L = my_net['parents'][str(net_L)][0]
            else:
                net_pa_L = []

            net_Ch_L = my_net['children'][str(net_L)][0]
            # print('latent variable: ')
            # print(net_L)
            # print(num)
            # print(net_list_L)
            # print('latent variable - value of VQVAE: ')
            # print(VQVAE.list_index[n].reshape(-1).tolist())
            df.loc[:, str(net_L)] = VQVAE.list_index[n].reshape(-1).tolist()
            # print(net.edges())

            states = net.states
            # print(net.states)

            remove_nodes = []
            # remove_edges = []
            for V in net_list_L:
                if V == str(net_L):
                    states[V] = [i for i in range(para['k_VQVAE'][n])]
                    pass
                else:
                    # df.loc[:, V] = 0
                    # # remove_edges.append((net_pa_L, V))
                    # remove_edges.append((V, net_Ch_L))
                    # states[V] = [0]

                    remove_nodes.append(V)
                    del df[V]
                    del states[V]
            # print(df)
            # print(states)
            # print(remove_edges)
            # net.remove_edges_from(remove_edges)
            # print(remove_nodes)
            net.remove_nodes_from(remove_nodes)
            # print(net.edges())
            net.fit(df, state_names=states)

        # for cpd in net.get_cpds():
        #     print(cpd)

        BIF = BIFWriter(net)
        path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE' + '.bif'
        print(path_VQVAE_bif)
        BIF.write_bif(path_VQVAE_bif)

        from func.excel import save_excel

        latent_vars = []
        path_VQVAE_data = VQVAE_path + para['data_file'] + '_VQVAE' + '.xlsx'
        print(path_VQVAE_data)
        save_excel(latent_vars, df, path_VQVAE_data)

        latent_vars = net_list_L[0]
        path_VQVAE_data = VQVAE_path + para['data_file'] + '_VQVAE_L' + '.xlsx'
        print(path_VQVAE_data)
        save_excel(latent_vars, df, path_VQVAE_data)




        ################################################################################################################
        ################################################################################################################
        # ts = time.time()
        # BNML = structure_train_process(BNML, para, input, [U_index, I_index, R])
        # ################################################################################################################
        # ################################################################################################################
        # print('------Structure learning has been done!!--------------')
        # print("time=", "{:.4f}".format(time.time() - ts))
        # print("time=", "{:.4f}".format(time.time() - tl))
        # print("BNML.edges:")
        # print(BNML.edges)
        # for i in range(0, BNML.num_nodes):
        #     BNML.output_CPD_node_i(i, para)
        # print()
        # # print("list_BIC_loglikelihood:")
        # # print(BNML.list_BIC_loglikelihood)
        # # print("list_BIC_penalty:")
        # # print(BNML.list_BIC_penalty)
        # # print("list_BIC:")
        # # print(BNML.list_BIC)
        # # print(BNML.list_NPN)
        #
        # for i in range(0, BNML.num_nodes):
        #     BNML.output_CPD_node_i(i, para)
        #
        # if para['Save_model'] == True:
        #     BNML.save_model(para)
    ################################################################################################################
    ################################################################################################################