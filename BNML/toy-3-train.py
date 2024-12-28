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
from func.find_cpd import find_cpt, change_cpt


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
    para_init.read_cpd_rould = None
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    num_latent = para['num_latent_variables']

    path = os.path.dirname(os.path.dirname(__file__)) + "\\data\\" + para['data_file'] + '\\'
    VQVAE_path = path + 'VQVAE\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'

    net_bif = path
    # data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + para['file_name_train']
    # fulldata_path = path + 'fulldata\\' + str(para['num_sample']) + '\\'
    fulldata_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'
    missingdata_path = path + 'missingdata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\'


    path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE' + '.bif'
    path_VQVAE_data = VQVAE_path + para['data_file'] + '_VQVAE' + '.xlsx'
    ################################################################################################################
    ################################################################################################################
    print("0: Reading Network . . . ")
    reader = my_BIFReader(VQVAE_path + para['data_file'] + '_VQVAE_BNML_2.bif')
    # reader = my_BIFReader(VQVAE_path + para['data_file'] + '_VQVAE_BNML_1.bif')
    my_net_VQVAE = reader.my_model()
    # for v in my_net_VQVAE['V']:
    #     print(my_net_VQVAE['learn_cpds_df'][v])

    # print("1: Getting data from records . . . ")
    # df_VQVAE = my_bayesnet.read_excel_data(VQVAE_path + para['data_file'] + '_VQVAE.xlsx')
    # print(df_VQVAE)
    # exit(0)
    ################################################################################################################
    ################################################################################################################
    print("0: Reading Network . . . ")
    print(fulldata_path)
    reader = my_BIFReader(fulldata_path + para['data_file'] + '.bif')
    my_net = reader.my_model()
    # for v in my_net['V']:
    #     print(my_net['learn_cpds_df'][v])

    # print("1: Getting data from records . . . ")
    # df = my_bayesnet.read_excel_data(data_file_path)
    # print(df)
    # exit(0)
    ################################################################################################################
    ################################################################################################################
    BNML = load_BN_files(para, 'D', VQVAE_path)

    # for i in range(0, BNML.num_nodes):
    #     output_m, output_s = BNML.output_CPD_node_i(i, para)
    #     print(output_m)
    # exit(0)

    if "0" in BNML.list_node_label:
        print("------------loading VQVAE!-------------------")
        VQVAE = AEs.load_AE(para, "VQVAE-files", VQVAE_path).to(para['device'])
    # print(BNML.num_nodes)

    # exit(0)
    ################################################################################################################
    ################################################################################################################
    for v in my_net['V']:
        # print(v)
        cpt_net = my_net['learn_cpds_df'][v]
        if int(v) in para['list_node_index_VQVAE']: #
            V_BNML = para['list_node_index_VQVAE'].index(int(v))

            if BNML.list_dependent_latent_node(V_BNML) == []: # 当前节点为与隐变量无关的节点
                my_net['learn_cpds_df'][v] = my_net_VQVAE['learn_cpds_df'][v]

            else: # 当前节点为与隐变量有关的节点
                if BNML.list_latent_variables[V_BNML] == 1: # 当前节点为隐变量
                    pass
                elif BNML.list_latent_variables[V_BNML] == 0: # 当前节点为隐变量的儿子，且为显变量
                    L_BNML = BNML.list_parent_nodes(V_BNML)[0]
                    # print('L_BNML: ' + str(L_BNML))
                    n = int(para['list_node_label'][L_BNML])
                    # print('index_VQVAE: ' + str(n))
                    net_L = para['VQVAE_node_label'].index(str(n)) + 1
                    # print('net_L: ' + str(net_L))
                    num = para['VQVAE_node_label'].count(str(n))
                    net_list_L = [str(l) for l in range(net_L, net_L + num)]
                    # print('net_list_L: ' + str(net_list_L))

                    node_cardinalities_combinations = para['node_cardinalities_combinations'][n]
                    sum_cardinalities_nodes = para['sum_node_cardinalities'][n]

                    # print(cpt_net)
                    # print(my_net_VQVAE['learn_cpds_df'][v])

                    num_paras_V_net = cpt_net.shape[0]
                    features_L = [None for i in range(para['num_VQVAE'])]
                    for index in range(num_paras_V_net):
                        # print('index: ' + str(index))
                        # 生成feature
                        net_v_value = cpt_net.loc[index, v]
                        # print('net_v_value: ' + str(net_v_value))
                        net_L_values = []
                        for L in net_list_L:
                            net_L_values.append(cpt_net.loc[index, L])
                        # print('net_L_values: ' + str(net_L_values))

                        data1 = torch.LongTensor(1, 1).zero_().to(para['device'])
                        # print(data1)
                        s = sum(np.multiply(node_cardinalities_combinations, net_L_values)) + net_L_values[0]
                        # print(s)
                        data1[0] = s
                        # print(data1)
                        data = torch.Tensor(1, sum_cardinalities_nodes).zero_().to(para['device'])
                        data.scatter_(1, data1, 1)
                        # print(data)
                        features_L[n] = data
                        Z, z_e, z_q, emb, X_pred = VQVAE(features_L, n)
                        net_VQVAE_L_values = VQVAE.list_index[n].reshape(-1).tolist()[0]
                        net_VQVAE_p = find_cpt(my_net_VQVAE, [str(net_L)], [net_VQVAE_L_values], v, net_v_value)
                        # print(net_VQVAE_p)

                        change_cpt(my_net, net_list_L, net_L_values, v, net_v_value, net_VQVAE_p)
                        # print(my_net['learn_cpds_df'][v])
                        # exit(0)
        else:
            pass


    # for v in my_net['V']:
    #     print(my_net['learn_cpds_df'][v])

    ################################################################################################################
    ################################################################################################################
    BIF = my_BIFWriter(my_net)
    path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE_BNML_back' + '.bif'
    print(path_VQVAE_bif)
    BIF.write_bif(path_VQVAE_bif)

    ################################################################################################################
    ################################################################################################################
