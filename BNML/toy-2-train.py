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
from my_model.BN import BN, load_BN, load_BN_files
from data.loaddata import load_data, process_features, load_BN_trainset
from my_model import AEs
from data.loaddata import load_tarin_dl
from learning.toy_All_AE_NPN_leraning_process_D import All_AE_NPN_learning
import torch.nn as nn
import torch.nn.functional as F
import os
from BN.my_BIF import BIFReader as my_BIFReader
from BN.my_BIF import BIFWriter as my_BIFWriter
from pgmpy.readwrite.BIF import BIFReader, BIFWriter
from BN import my_bayesnet
from func.find_cpd import change_all_cpt
from BN.get_missing_index import get_missing_index

loss_func_MSE = nn.MSELoss()

if __name__ == '__main__':
    para = init.para
    para = init.func(para, 'D')
    # print(para)
    # exit(0)
    if para['device'] == "gpu":
        para['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(para['list_edges'])
    # print(para)
    # exit(0)

    SEED = para_init.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    para_init.read_cpd_rould = 16
    # para_init.read_cpd_rould = 1

    num_latent = para['num_latent_variables']

    path = os.path.dirname(os.path.dirname(__file__)) + "\\data\\" + para['data_file'] + '\\'
    VQVAE_path = path + 'VQVAE\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'

    net_bif = path
    # data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + para['file_name_train']
    data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\' + para['file_name_train']

    # missingdata_path = path + 'missingdata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'


    path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE' + '.bif'
    path_VQVAE_data = VQVAE_path + para['data_file'] + '_VQVAE' + '.xlsx'
    path_VQVAE_L_data = VQVAE_path + para['data_file'] + '_VQVAE_L' + '.xlsx'
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
        print('BNML.edges[0]:')
        print(BNML.edges[0])
        print(BNML.edges[1])
        print(BNML.edges[2])
        # print(BNML.edges[3])
        # V_net = 4
        # print('V_net: ' + str(V_net))
        # V_BNML = para['list_node_index_VQVAE'].index(int(V_net))
        # print('V_BNML: ' + str(V_BNML))
        # print(BNML.list_parents(V_BNML))
        # print(BNML.list_cardinalities)


        # print('BNML.constraint_may_edges:')
        # print(BNML.constraint_may_edges)
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        features_L = None
        VQVAE = None
        print("---------------加载特征数据！---------------------")
        features_origin = load_data(para, 'feature', path_VQVAE_data)
        # print(features_origin)
        print(features_origin[0])
        print(features_origin[1])
        print(features_origin[2])
        # exit(0)
        # features_L = process_features(para, BNML, features_origin)
        print(features_L)
        # print(features[0].shape) # torch.Size([12, 4])
        # print(features_origin[0])
        # exit(0)
        if "0" in BNML.list_node_label:
            print("------------loading VQVAE!-------------------")
            VQVAE = AEs.load_AE(para, "VQVAE-files", VQVAE_path).to(para['device'])
        #     print(VQVAE)
        #     print(VQVAE.list_index[0])
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
        print(tarin_dl_tensor_list[0])
        print(tarin_dl_tensor_list[0][0][0])
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        print("0: Reading Network . . . ")
        reader = my_BIFReader(path_VQVAE_bif)
        my_net = reader.my_model()
        # print(my_net['learn_cpds_df'])

        print("1: Getting data from records . . . ")
        df = my_bayesnet.read_excel_data(path_VQVAE_L_data)
        print(df)
        # exit(0)

        print("3: Getting missing data indexes . . . ")
        mis_index, mis_var = get_missing_index(df)
        print(mis_index)
        print(mis_var)

        # Initialise parameters
        print("2: Initialising parameters . . . ")
        my_net = my_bayesnet.my_init_param(df, my_net, mis_var)
        # print(my_net['learn_cpds_df'])

        # for v in net['V']:
        #     print(net['learn_cpds_df'][v])
        #     if v in mis_var:
        #         print(states[V])
        #         net['learn_cpds_df'][v]['p'] = 1.0 / states[V]
        #     print(net['learn_cpds_df'][v])

        # print(my_net['V'])
        # for v in my_net['V']:
        #     # print(my_net['cpds'][v])
        #     # print(my_net['cpds_df'][v])
        #     print(my_net['learn_cpds_df'][v])
        # exit(0)
        ################################################################################################################
        ################################################################################################################
        print("---------------BNML pre-learning---------------------")
        try:
            BNML = load_BN_files(para, 'pre', VQVAE_path)
            # print('BNML.edges[0]:')
            # print(BNML.edges[0])
            # print(BNML.edges[1])
            # print(BNML.edges[2])
            # print(BNML.edges[3])
            # V_net = 4
            # print('V_net: ' + str(V_net))
            # V_BNML = para['list_node_index_VQVAE'].index(int(V_net))
            # print('V_BNML: ' + str(V_BNML))
            # print(BNML.list_parents(V_BNML))
            # print(BNML.list_cardinalities)
        except:
            para['BN_max_convergence_iter_num'] = 250
            # para['BN_parameter_learning_max_iter_num'] = 5000
            para['BN_parameter_learning_max_iter_num'] = 1500
            # para['BN_parameter_learning_max_iter_num'] = 1000
            t_pre_strat = time.time()
            BNML, _ = All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features_L, df, VQVAE, False, my_net, curr_iter = 0)
            t_pre_stop = time.time()
            print("pre ITERATION Time: " + str(round((t_pre_stop - t_pre_strat), 4)))
            change_all_cpt(my_net, BNML, para)
            BNML.save_BN_files(para, 'pre', VQVAE_path)
        # exit(0)
        para['BN_max_convergence_iter_num'] = 50
        # if para_init.bn == 'child' and para_init.num_latent == 4:
        #     para['BN_max_convergence_iter_num'] = 100
        para['BN_parameter_learning_max_iter_num'] = 1000
        ################################################################################################################
        ################################################################################################################
        print("---------------BNML learning---------------------")
        curr_iter = 1
        # time_i = time.time()

        list_em_time = []
        start_time = time.time()
        while True:
            print("ITERATION #" + str(curr_iter))

            prev_cpts = []  # pre_cpts denotes the initialized cpt by init_param method
            for X in my_net['V']:
                V_BNML = para['list_node_index_VQVAE'].index(int(X))
                # if BNML.list_dependent_latent_node(V_BNML) == []:
                #     continue
                prev_cpts.append(np.array(list(my_net['learn_cpds_df'][X]['p'])))
            #     print(my_net['learn_cpds_df'][X])
            #     print(my_net['learn_cpds_df'][X]['p'])
            # print('prev_cpts:\n', prev_cpts)
            # exit(0)

            # for i in range(0, BNML.num_nodes):
            #     output_m, output_s = BNML.output_CPD_node_i(i, para)
            #     print(output_m)

            # change_all_cpt(my_net, BNML, para)

            # for X in my_net['V']:
            #     prev_cpts.append(np.array(list(my_net['learn_cpds_df'][X]['p'])))
            #     print(my_net['learn_cpds_df'][X])
            #     print(my_net['learn_cpds_df'][X]['p'])
            #     print('prev_cpts:\n', prev_cpts)
            # exit(0)

            t_start = time.time()
            BNML, _ = All_AE_NPN_learning(BNML, para, tarin_dl_tensor_list, test_dl, features_L, df, VQVAE, False, my_net, curr_iter)
            t_stop = time.time()
            print("ITERATION Time: " + str(round((t_stop - t_start), 4)))
            list_em_time.append(t_stop - t_start)

            change_all_cpt(my_net, BNML, para, curr_iter)

            # for i in range(0, BNML.num_nodes):
            #     output_m, output_s = BNML.output_CPD_node_i(i, para)
            #     print(output_m)
            # exit(0)

            new_cpts = []  # new_cpts denotes the learned cpt by EM algorithm
            for X in my_net['V']:
                V_BNML = para['list_node_index_VQVAE'].index(int(X))
                # if BNML.list_dependent_latent_node(V_BNML) == []:
                #     continue
                new_cpts.append(np.array(list(my_net['learn_cpds_df'][X]['p'])))
                # print(my_net['learn_cpds_df'][X])
                # print(my_net['learn_cpds_df'][X]['p'])
            # print('new_cpts:\n', new_cpts)
            # exit(0)
            diffs = []
            for i in range(len(prev_cpts)):
                max_diff = max(abs(np.subtract(prev_cpts[i], new_cpts[i])))  # subtract denotes the - operation
                diffs.append(max_diff)
            # print(diffs)
            delta = np.round(max(diffs), 4)
            # print(delta)
            # exit(0)
            # time_f = time.time()
            print("Delta: " + str(delta))

            # if (time_f - time_i) > 660:
            #     print("OVER TIME. . . . ")
            #     break

            BIF = my_BIFWriter(my_net)
            path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE_BNML_' + str(curr_iter) + '.bif'
            # print(path_VQVAE_bif)
            BIF.write_bif(path_VQVAE_bif)

            if delta <= para_init.threshold_delta or curr_iter == para_init.max_iter:
                break
            curr_iter += 1
        print("Converged in (" + str(curr_iter) + ") iterations")
        # exit(0)
        end_time = time.time()
        print('em_time:', sum(list_em_time))
        print('runtime:', end_time - start_time)


        for i in range(0, BNML.num_nodes):
            BNML.list_BIC_penalty[i] = BNML.independent_parameters_node_i(i) / 2 * np.log(para['train_size'])
            BNML.list_BIC[i] = BNML.list_BIC_loglikelihood[i] - BNML.list_BIC_penalty[i]

        if para['Save_model'] == True:
            BNML.save_BN_files(para, 'D', VQVAE_path)

        ################################################################################################################
        ################################################################################################################
        print("list_BIC_loglikelihood:")
        print(BNML.list_BIC_loglikelihood)
        print("list_BIC_penalty:")
        print(BNML.list_BIC_penalty)
        print("list_BIC:")
        print(BNML.list_BIC)
        print()

        # for i in range(0, BNML.num_nodes):
        #     output_m, output_s  = BNML.output_CPD_node_i(i, para)
        #     print(output_m)
        # exit(0)
        ################################################################################################################
        ################################################################################################################


        ################################################################################################################
        ################################################################################################################

    ################################################################################################################
    ################################################################################################################