from pgmpy.inference import VariableElimination, ApproxInference
from sklearn.metrics import confusion_matrix
from init_para import child_L2_init_para as init
import torch
import numpy as np
import time
import para_init
from my_model.BN import BN, load_BN, load_BN_files
import os
from BN.my_BIF import BIFReader as my_BIFReader
from BN.my_BIF import BIFWriter as my_BIFWriter
from pgmpy.readwrite.BIF import BIFReader, BIFWriter
from BN import my_bayesnet

import torch
import numpy as np
import time
from data.loaddata import load_data, process_features, load_BN_trainset
from data.loaddata import load_tarin_dl
from prediction.MAP import NPN_MAP_process



#################################### 注意 ############################################
read_cpd_rould_decimal = None
read_cpd_rould = None
#################################### 注意 ############################################
def average(lst):
    return sum(lst) / len(lst)


def cm_metrix(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # print(TP)
    # print(FP)
    # print(TN)
    # print(FN)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # print(TPR)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP)
    # # print(TNR)
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # print(PPV)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # print(NPV)
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # print(FPR)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # print(FNR)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # print(FDR)

    precision = (TP+0.01) / (TP+FP+0.01)  # 查准率
    recall = (TP+0.01) / (TP+FN+0.01)  # 查全率
    F1 = (2 * precision * recall + 0.01) / (precision + recall+0.01)
    # print(precision)
    # print(recall)
    # print(F1)

    # return TPR, TNR, PPV, NPV, FPR, FNR, FDR, precision, recall, F1
    # return precision, recall, F1

    return average(precision), average(recall), average(F1)


def infer(df, network, type_infer = None):
    path_inference = os.path.dirname(__file__) + '\\' + 'inference\\'
    # print(path_inference)
    # exit(0)
    file = open(path_inference + para_init.bn + "_" + type_infer + ".txt", "w")


    if type_infer == 'VE':
        print('-----------VE---------------')
        inference = VariableElimination(network)
    # elif type_infer == 'BP':
    #     print('-----------BP---------------')
    #     inference = BeliefPropagation(network)
    elif type_infer == 'FS':
        print('-----------FS---------------')
        inference = ApproxInference(network, type_infer, para_init.n_samples)
    elif type_infer == 'RS':
        print('-----------RS---------------')
        inference = ApproxInference(network, type_infer, para_init.n_samples)
    else:
        print('type_infer is error!!!!!')
        exit(0)

    y_true = []
    y_pred = []
    num = df.shape[0]
    t1 = time.time()
    for n in range(num):
        variables = [str(i) for i in para['list_search_node_temp']]
        evidence = {str(i): str(df.iloc[n][str(i)]) for i in para['list_evidence_node_temp']}
        # print(variables)
        # print(evidence)
        y_true.append(df.iloc[n][str(variables[0])])
        if type_infer == 'VE':
            phi_query = inference.query(variables=variables, evidence=evidence, show_progress=False)
        elif type_infer == 'FS' or type_infer == 'RS':
            phi_query = inference.query(variables=variables, n_samples=para_init.num_sample, evidence=evidence, show_progress=False)
        narray_p = phi_query.values
        # print(narray_p)

        line = str(n+1)
        for p in narray_p:
            line += ',' + str(p)
        line += '\n'
        file.write(line)

        y = np.argmax(narray_p)
        # print(y)
        y_pred.append(y)
        # exit(0)
        # if n+1 == int(para_init.num_infer/16):
        #     print(str(int(para_init.num_infer/16)) + ' time: ' + str(time.time()-t1))
        # if n+1 == int(para_init.num_infer/8):
        #     print(str(int(para_init.num_infer/8)) + ' time: ' + str(time.time()-t1))
        # if n+1 == int(para_init.num_infer/4):
        #     print(str(int(para_init.num_infer/4)) + ' time: ' + str(time.time()-t1))
        # if n+1 == int(para_init.num_infer/2):
        #     print(str(int(para_init.num_infer/2)) + ' time: ' + str(time.time()-t1))
        if n+1 == int(para_init.num_infer):
            print(str(int(para_init.num_infer)) + ' time: ' + str(time.time()-t1))
            break


    t2 = time.time()
    print(type_infer + ' time: ' + str(t2-t1))
    # print(y_true)
    # print(y_pred)
    file.close()
    # exit(0)
    return y_true, y_pred

####################################################################################################################
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
    data_file_path = path + 'fulldata\\' + str(para['num_sample']) + '\\' + str(num_latent) + '\\' + para['file_name_train']

    path_VQVAE_bif = VQVAE_path + para['data_file'] + '_VQVAE_BNML_1' + '.bif'
    path_VQVAE_data = VQVAE_path + para['data_file'] + '_VQVAE' + '.xlsx'
    ################################################################################################################
    ################################################################################################################
    print("0: Reading Network . . . ")
    print(path_VQVAE_bif)
    reader = BIFReader(path_VQVAE_bif, n_jobs=1)
    network = reader.get_model()
    print(network.nodes())
    print(network.get_cpds(str(4)))
    # exit(0)

    print("1: Loading Data . . . ")
    # df = my_bayesnet.read_excel_data(path_VQVAE_data)
    df = my_bayesnet.read_excel_data(data_file_path)
    print(df)
    # exit(0)

    print('list_evidence_node_temp:')
    print(para['list_evidence_node_temp'])
    print('list_search_node_temp:')
    print(para['list_search_node_temp'])
    print('list_evidence_node:')
    print(para['list_evidence_node'])
    print('list_search_node:')
    print(para['list_search_node'])
    # exit(0)
    ################################################################################################################
    ##################################################    VE      ##################################################
    y_true, y_pred_VE = infer(df, network, 'VE')
    # print(y_true)
    # print(y_pred_VE)
    # cm = confusion_matrix(y_true, y_pred_VE)
    # precision, recall, F1 = cm_metrix(cm)
    # print('VE precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))
    # TPR, TNR, PPV, NPV, FPR, FNR, FDR, precision, recall, F1 = cm_metrix(cm)
    # print('VE , TPR: {:.4f}, TNR: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, FPR: {:.4f}, FNR: {:.4f}, FDR: {:.4f}'.format(TPR, TNR, PPV, NPV, FPR, FNR, FDR))
    exit(0)

    # para_init.n_samples = 1000
    # print('n_samples: ' + str(para_init.n_samples))
    y_true, y_pred_FS = infer(df, network, 'FS')
    # print(y_true)
    # print(y_pred_FS)
    # cm = confusion_matrix(y_pred_VE, y_pred_FS)
    # precision, recall, F1 = cm_metrix(cm)
    # print('FS precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))
    # cm = confusion_matrix(y_true, y_pred_FS)
    # precision, recall, F1 = cm_metrix(cm)
    # print('FS precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))

    y_true, y_pred_RS = infer(df, network, 'RS')
    # print(y_true)
    # print(y_pred_RS)
    # cm = confusion_matrix(y_pred_VE, y_pred_RS)
    # precision, recall, F1 = cm_metrix(cm)
    # print('RS precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))
    # cm = confusion_matrix(y_true, y_pred_RS)
    # precision, recall, F1 = cm_metrix(cm)
    # print('FS precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))

    ################################################################################################################
    ##################################################    BNML     #################################################
    BNML = load_BN_files(para, 'D', VQVAE_path)
    BNML.list_evidence_node = para['list_evidence_node']
    BNML.list_search_node = para['list_search_node']

    # for i in range(0, BNML.num_nodes):
    #     output_m, output_s = BNML.output_CPD_node_i(i, para)
    #     print('CPD of node: ' + str(i+1))
    #     print(output_m)
    # exit(0)
    # if "0" in BNML.list_node_label:
    #     print("------------loading VQVAE!-------------------")
    #     VQVAE = AEs.load_AE(para, "VQVAE-files", VQVAE_path).to(para['device'])
    # print(BNML.num_nodes)
    # exit(0)

    features_L = None
    print("---------------加载BN训练数据！---------------------")
    train_dl, test_dl, trainLen, testLen = load_BN_trainset(para, test_batch_size=None, shuffle=False,
                                                            data_file_path=data_file_path)
    print('trainLen:' + str(trainLen))
    print('testLen:' + str(testLen))
    para['train_size'] = trainLen

    tarin_dl_tensor_list = []
    for batch_idx, data in enumerate(train_dl):  # batch_idx: 0~n
        input, index, I_index, R = load_tarin_dl(data, para, BNML, features_L)
        tarin_dl_tensor_list.append([input, index, I_index, R])
    # print(tarin_dl_tensor_list[0])
    # print(tarin_dl_tensor_list[0][0][0])
    # exit(0)
    #########################################################################################################
    #########################################################################################################
    para['EM'] = 'False'
    para['prior'] = 'False'
    t_map = time.time()
    print("---------------MAP start---------------------")
    list_not_evidence_var, list_not_evidence_var_Norm, list_not_evidence_var_index = NPN_MAP_process(BNML, para, tarin_dl_tensor_list)
    print("MAP time=", "{:.4f}".format(time.time() - t_map))

    # print("---------------MAP result---------------------")
    print('list_not_evidence_var:')
    for i in list_not_evidence_var:
        print(i)
    print('list_not_evidence_var_Norm:')
    for i in list_not_evidence_var_Norm:
        print(i)
    print('list_not_evidence_var_index:')
    print(list_not_evidence_var_index)
    # BNML.list_evidence_node:
    # [1, 4, 7, 8]
    # BNML.list_search_node:
    # [2, 3, 5]

    # print("---------------MAP precision---------------------")
    for i in BNML.list_search_node:
        # print(i)
        j = i - 1
        correct = 0
        index = list_not_evidence_var_index[j] # [-1, 0, 1, -1, 2, 3, -1, -1]
        # print(index)
        y_pred = None
        target = input[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
        if BNML.list_node_type[index] == 'C':
            pred = list_not_evidence_var[index]
            MSE = torch.pow(pred - target, 2).sum()
            RMSE = torch.sqrt(MSE, 2)
            print('Node: {}, MSE: {:.4f}, RMSE: {:.4f}'.format(i, MSE, RMSE))
        elif BNML.list_node_type[index] == 'D':
            pred = list_not_evidence_var_Norm[index]
            # print('pred:')
            # print(pred)


            list_pred = pred.cpu().detach().numpy().tolist()
            # print(list_pred)
            path_inference = os.path.dirname(__file__) + '\\' + 'inference\\'
            file = open(path_inference + para_init.bn + "_" +'BNML' + ".txt", "w")
            for n in range(len(list_pred)):
                narray_p = list_pred[n]
                # print(narray_p)
                line = str(n + 1)
                for p in narray_p:
                    line += ',' + str(p)
                line += '\n'
                file.write(line)
            file.close()


            # pred = pred.data.max(axis=1, keepdim=True)[1] + 1 # get the index of the max log-probability
            # print('pred:')
            # print(pred)
            # print('target:')
            # print(target)
            # target = target.data.max(axis=1, keepdim=True)[1] + 1
            # print('target:')
            # print(target)
            # correct += pred.eq(target.view_as(pred)).long().cpu().sum()
            # print('Node: {}, Accuracy: {}/{} ({:.2f}%)'.format(i, correct, len(input), 100. * correct / len(input)))
            # exit(0)

            pred = pred.data.max(axis=1, keepdim=True)[1]
            # print('pred:')
            # print(pred)
            y_pred_BNML = pred.reshape(-1).tolist()
            # print(y_pred)

            # print(y_true)
            # print(y_pred)
            # cm = confusion_matrix(y_pred_VE, y_pred_BNML)
            # precision, recall, F1 = cm_metrix(cm)
            # print('BNML precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))
            # cm = confusion_matrix(y_true, y_pred_BNML)
            # precision, recall, F1 = cm_metrix(cm)
            # print('FS precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, F1))






