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
from metrics import MSE, MAE, KL

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

    print_float_round = para_init.print_float_round
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
    para_init.read_cpd_rould = 1
    print("0: Reading Network . . . ")
    print(fulldata_path)
    reader = my_BIFReader(fulldata_path + para['data_file'] + '.bif')
    my_net_origin_round1 = reader.my_model()
    for v in my_net_origin_round1['V']:
        print(my_net_origin_round1['learn_cpds_df'][v])
    print()
    ################################################################################################################
    para_init.read_cpd_rould = None
    print("0: Reading Network . . . ")
    print(fulldata_path)
    reader = my_BIFReader(fulldata_path + para['data_file'] + '.bif')
    my_net_origin = reader.my_model()
    for v in my_net_origin['V']:
        print(my_net_origin['learn_cpds_df'][v])
    print()
    ################################################################################################################
    ################################################################################################################
    def MSE_KL(my_net_origin, my_net_learn):
        ori_cpts = []
        cur_cpts = []
        for X in my_net_origin['V']:
            # print('current: ' + X)
            # print(my_net_origin['learn_cpds_df'][X])
            ori_cpt = list(my_net_origin['learn_cpds_df'][X]['p'])
            # print(ori_cpt)
            ori_cpt = [float(i) for i in ori_cpt]
            # print(ori_cpt)
            # exit(0)

            # print(my_net_learn['learn_cpds_df'][X])
            cur_cpt = list(my_net_learn['learn_cpds_df'][X]['p'])
            # print(cur_cpt)
            cur_cpt = [float(i) for i in cur_cpt]
            # print(cur_cpt)

            # print('V:', X +
            #       '   MSE: ' + str(round(MSE(np.array(ori_cpt), np.array(cur_cpt)), print_float_round)) +
            #       '   MAE: ' + str(round(MAE(np.array(ori_cpt), np.array(cur_cpt)), print_float_round)) +
            #       '   KL_o_c: ' + str(round(KL(ori_cpt, cur_cpt), print_float_round)) +
            #       '   KL_c_o: ' + str(round(KL(cur_cpt, ori_cpt), print_float_round))
            #       )
            # ori_cpts.append(ori_cpt)
            # cur_cpts.append(cur_cpt)

            car = len(my_net_origin['states'][X])
            temp_ori_cpt = []
            temp_cur_cpt = []
            if len(ori_cpt) != len(cur_cpt):
                print('Node: ' + str(X) + '. num_ori_cpt is not equal to num_cur_cpt')
                exit(0)
            else:
                for i in range(len(ori_cpt)):
                    if abs(ori_cpt[i] - 1.0 / car) > 0.0001:
                        temp_ori_cpt.append(ori_cpt[i])
                        temp_cur_cpt.append(cur_cpt[i])
                    else:
                        temp_ori_cpt.append(1.0 / car)
                        temp_cur_cpt.append(1.0 / car)


            # if X == '4':
            #     print(temp_ori_cpt)
            #     print(temp_cur_cpt)

            print('V:', X +
                  '   MSE: ' + str(round(MSE(np.array(temp_ori_cpt), np.array(temp_cur_cpt)), print_float_round)) +
                  '   MAE: ' + str(round(MAE(np.array(temp_ori_cpt), np.array(temp_cur_cpt)), print_float_round)) +
                  '   KL_o_c: ' + str(round(KL(temp_ori_cpt, temp_cur_cpt, car), print_float_round))
                  # '   KL_c_o: ' + str(round(KL(temp_cur_cpt, temp_ori_cpt), print_float_round))
                  )
            ori_cpts.append(temp_ori_cpt)
            cur_cpts.append(temp_cur_cpt)

        # exit(0)

        # ori_cpts = [i for j in ori_cpts for i in j]
        # cur_cpts = [i for j in cur_cpts for i in j]
        # print('MSE: ' + str(round(MSE(np.array(ori_cpts), np.array(cur_cpts)), print_float_round)) +
        #       '   MAE: ' + str(round(MAE(np.array(ori_cpts), np.array(cur_cpts)), print_float_round)) +
        #       '   KL_o_c: ' + str(round(KL(ori_cpts, cur_cpts), print_float_round)) +
        #       '   KL_c_o: ' + str(round(KL(cur_cpts, ori_cpts), print_float_round))
        #       )
    ################################################################################################################
    ################################################################################################################
    # print()
    # print("round1:")
    # reader = my_BIFReader(VQVAE_path + para['data_file'] + '_VQVAE_BNML_back.bif')
    # my_net_learn = reader.my_model()
    # MSE_KL(my_net_origin, my_net_origin_round1)

    print()
    print("EM:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_EM.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)

    print()
    print("IEM:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_IEM.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)

    print()
    print("DEBN:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_DEBN.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)

    # # print()
    # # print("DEBN_robustness:")
    # # reader = my_BIFReader(missingdata_path + para['data_file'] + '_DEBN_robustness.bif')
    # # my_net_learn = reader.my_model()
    # # MSE_KL(my_net_origin, my_net_learn)
    #
    print()
    print("TRIP:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_TRIP.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)

    print()
    print("RPL:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_RPL.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)
    #
    print()
    print("ARPL:")
    reader = my_BIFReader(missingdata_path + para['data_file'] + '_ARPL.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)


    print()
    print("VQVAE:")
    reader = my_BIFReader(VQVAE_path + para['data_file'] + '_VQVAE_BNML_back.bif')
    my_net_learn = reader.my_model()
    MSE_KL(my_net_origin, my_net_learn)
    # for v in my_net_learn['V']:
    #     print(my_net_learn['learn_cpds_df'][v])
    # print()





    