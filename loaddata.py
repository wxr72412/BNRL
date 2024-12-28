import numpy as np
import pandas as pd
import os
from torch.utils.data import random_split
import torch
from scipy.sparse import coo_matrix
from func.preprocessing import *

def load_data(para, data_type = None, data_file_path = None):
    project_path = os.getcwd()
    data_path = project_path + '\\data\\'
    # print(project_path) # C:\Users\89647\Desktop\BNML-20211216\medical_NPN
    # print(data_path) # C:\Users\89647\Desktop\BNML-20211216\medical_NPN\data\
    # print(data_type)
    # exit(0)
    if para['data_file'] == 'test':
        if data_type == "train":
            return pd.read_table(data_path + para['data_file'] + '\\' + para['file_name_train'], sep='::', engine='python', header=None).values
        elif data_type == "test":
            return pd.read_table(data_path + para['data_file'] + '\\' + para['file_name_test'], sep='::', engine='python', header=None).values
        else:
            raise NotImplementedError

    elif para['data_file'] == 'chest_clinic':
        if data_type == "train":
            print(data_path + para['data_file'] + '\\' + para['file_name_train'])
            return pd.read_table(data_path + para['data_file'] + '\\' + para['file_name_train'], sep=',', engine='python', header=None).values
        elif data_type == "test":
            # print(data_path + para['data_file'] + '\\' + file_name + para['file_name_2'])
            return pd.read_table(data_path + para['data_file'] + '\\' + para['file_name_test'], sep=',', engine='python', header=None).values
        else:
            raise NotImplementedError

    elif para['data_file'] == 'cora':
        if data_type == "feature":
            data_path += para['data_file'] + '\\' + para['file_name_feature']
            return pd.read_table(data_path, sep=' ', engine='python', header=None).values
        elif data_type == 'adj':
            data_path += para['data_file'] + '\\' + para['file_name_adj']
            return pd.read_table(data_path, sep=' ', engine='python', header=None).values

    elif para['data_file'] == 'ml-1m':
        if data_type == "feature":
            data_path1 = data_path + para['data_file'] + '\\' + para['file_name_user_feature']
            data_path2 = data_path + para['data_file'] + '\\' + para['file_name_item_feature']
            return pd.read_table(data_path1, sep=',', engine='python', header=None).values, pd.read_table(data_path2, sep=',', engine='python', header=None).values
        elif data_type == 'adj':
            data_path += para['data_file'] + '\\' + para['file_name_adj']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        elif data_type == 'train':
            data_path += para['data_file'] + '\\' + para['file_name_train']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        elif data_type == 'test':
            data_path += para['data_file'] + '\\' + para['file_name_test']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        else:
            raise NotImplementedError

    elif para['data_file'] == 'dermatology' or para['data_file'] == 'bone-marrow':
        if data_type == "feature":
            data_path += para['data_file'] + '\\' + para['file_name_train']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        elif data_type == 'train':
            data_path += para['data_file'] + '\\' + para['file_name_train']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        elif data_type == 'test':
            data_path += para['data_file'] + '\\' + para['file_name_test']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        else:
            raise NotImplementedError
    elif para['data_file'] == 'dermatology-origin':
        if data_type == 'train':
            data_path += 'dermatology-origin' + '\\' + para['file_name_train']
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        else:
            raise NotImplementedError
    elif para['data_file'] == 'dermatology-NPN' or para['data_file'] == 'dermatology-NPN-BNML-C' or para['data_file'] == 'dermatology-NPN-BNML-D'\
            or para['data_file'] == 'bone-marrow-NPN' or para['data_file'] == 'bone-marrow-NPN-BNML-C':
        if data_type == 'train':
            data_path += para['file_name_train']
            print(data_path)
            return pd.read_table(data_path, sep=',', engine='python', header=None).values
        else:
            raise NotImplementedError
    elif para['data_file'] == 'toy1' or para['data_file'] == 'toy2' or para['data_file'] == 'child' or para['data_file'] == 'pigs' or para['data_file'] == 'water' or para['data_file'] == 'munin1':
        print(data_file_path)
        if data_type == "feature" or data_type == "train" or data_type == "test" :
            df = pd.DataFrame(pd.read_excel(data_file_path, index_col=0))
            # print(type(df.index))
            # print(type(df.values))
            index = np.array(list(df.index))
            # print(index)
            values = np.array(df.values)
            # print(values)
            index_values = np.column_stack((index, values))
            # print(index_values)
            return index_values
        elif data_type == "df":
            df = pd.DataFrame(pd.read_excel(data_file_path, index_col=0))
            return df
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def process_features(para, BNML, features):
    if para['data_file'] == 'cora':
        features = torch.FloatTensor(features).to(para['device'])
        features = features[:, 1:features.shape[1]]
    elif para['data_file'] == 'toy2' or para['data_file'] == 'child' or para['data_file'] == 'pigs' or para['data_file'] == 'water' or para['data_file'] == 'munin1':
        # print(features)
        # print()
        # exit(0)
        datas = []
        feature_temp = features[:, 1:]
        # print(feature_temp)
        # print(data)
        for n in range(para['num_VQVAE']):
            # print(n)
            index = para['VQVAE_node_label'].index(str(n))
            num = para['VQVAE_node_label'].count(str(n))
            # print(index)
            # print(num)
            feature = feature_temp[:, index:index+num]
            # print("feature")
            # print(feature)
            # print(feature.shape)

            # exit(0)
            ###############################################################################################################################################
            node_cardinalities_combinations = para['node_cardinalities_combinations'][n]
            sum_cardinalities_nodes = para['sum_node_cardinalities'][n]
            # print(node_cardinalities_combinations)
            # print(sum_cardinalities_nodes)

            data1 = torch.LongTensor(len(feature[:, 0]), 1).zero_().to(para['device'])
            # print(data1)
            for i in range(len(feature[:, 0])):
                # print(i)
                # print(feature[i])
                s = sum(np.multiply(node_cardinalities_combinations, feature[i])) + feature[i][0]
                # print(s)
                data1[i] = s
                # print(data1)
                # exit(0)
            # print(data1)
            data = torch.Tensor(len(feature[:, 0]), sum_cardinalities_nodes).zero_().to(para['device'])
            # print(data)
            # exit(0)
            data.scatter_(1, data1, 1)
            # print(data)
            ##########################################################################################################################################################
            # sum_cardinalities_nodes = sum(para['node_cardinalities'][n])
            # # print(para['node_cardinalities'][n])
            # # print(sum_cardinalities_nodes)
            # # exit(0)

            # data = torch.FloatTensor(len(feature[:, 0]), sum_cardinalities_nodes).zero_().to(para['device'])
            # for j in range(len(para['node_cardinalities'][n])):
            #     if para['node_type'][n][j] == 'D':
            #         xx = torch.FloatTensor(len(feature[:, 0]), para['node_cardinalities'][n][j]).zero_().scatter_(1, torch.LongTensor(feature[:, j].astype(int)).unsqueeze(1), 1)
            #         data[:, sum(para['node_cardinalities'][n][:j]):sum(para['node_cardinalities'][n][:j+1])] = xx
            # print(data.shape) # torch.Size([358, 88])
            # print(data)
            ###############################################################################################################################################
            # exit(0)
            datas.append(data)
        return datas



    elif para['data_file'] == 'dermatology':
        # print(features)
        # print()
        # exit(0)
        user_features = features[:, 1:13]
        item_features = features[:, 13:35]
        # print("user_features")
        # print(user_features)
        # print(user_features.shape) # (358, 12)
        # print("item_features")
        # print(item_features)
        # print(item_features.shape) # (358, 22)
        # exit(0)

        sum_cardinalities_user_nodes = 0
        for i in range(len(para['user_node_cardinalities'])):
            sum_cardinalities_user_nodes += para['user_node_cardinalities'][i]
        # print(para['user_node_cardinalities'])
        # print(sum_cardinalities_user_nodes)
        # exit(0)
        user_data = torch.FloatTensor(len(user_features[:, 0]), sum_cardinalities_user_nodes).zero_().to(para['device'])
        for j in range(len(para['user_node_cardinalities'])):
            if para['user_node_type'][j] == 'D':
                # print(user_features[:, j])
                # exit(0)
                xx = torch.FloatTensor(len(user_features[:, 0]), para['user_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(user_features[:, j].astype(int)).unsqueeze(1), 1)
                # print(xx)
                # exit(0)
                user_data[:, sum(para['user_node_cardinalities'][:j]):sum(para['user_node_cardinalities'][:j+1])] = xx
                # print(user_data)
                # exit(0)

        sum_cardinalities_item_nodes = 0
        for i in range(len(para['item_node_cardinalities'])):
            sum_cardinalities_item_nodes += para['item_node_cardinalities'][i]
        # print(para['item_node_cardinalities'])
        # print(sum_cardinalities_item_nodes)
        # exit(0)
        item_data = torch.FloatTensor(len(item_features[:, 0]), sum_cardinalities_item_nodes).zero_().to(para['device'])
        for j in range(len(para['item_node_cardinalities'])):
            if para['item_node_type'][j] == 'D':
                xx = torch.FloatTensor(len(item_features[:, 0]), para['item_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(item_features[:, j].astype(int)).unsqueeze(1), 1)
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j+1])] = xx
        # print(user_data.shape) # torch.Size([358, 118])
        # print(user_data[0])
        # for a in user_data[0]:
        #     print("{:.2f}".format(a.item()) + "  ", end='')
        # print()
        #
        # print(item_data.shape) # torch.Size([358, 88])
        # print(item_data[0])
        # for a in item_data[0]:
        #     print("{:.2f}".format(a.item()) + "  ", end='')
        # print()
        # exit(0)
        features = [user_data, item_data]



    elif para['data_file'] == 'bone-marrow':
        # print(features)
        # print()
        # exit(0)
        user_features = features[:, 1:27]
        item_features = features[:, 27:30]
        # print("user_features")
        # print(user_features)
        # print(user_features.shape) # (142, 26)
        # print("item_features")
        # print(item_features)
        # print(item_features.shape) # (142, 3)
        # exit(0)

        sum_cardinalities_user_nodes = 0
        for i in range(len(para['user_node_cardinalities'])):
            sum_cardinalities_user_nodes += para['user_node_cardinalities'][i]
        # print(para['user_node_cardinalities'])
        # print(sum_cardinalities_user_nodes)
        # exit(0)
        user_data = torch.FloatTensor(len(user_features[:, 0]), sum_cardinalities_user_nodes).zero_().to(para['device'])
        for j in range(len(para['user_node_cardinalities'])):
            if para['user_node_type'][j] == 'D':
                # print(user_features[:, j])
                # exit(0)
                xx = torch.FloatTensor(len(user_features[:, 0]), para['user_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(user_features[:, j].astype(int)).unsqueeze(1), 1)
                # print(xx)
                # exit(0)
                user_data[:, sum(para['user_node_cardinalities'][:j]):sum(para['user_node_cardinalities'][:j+1])] = xx
                # print(user_data)
                # exit(0)

        sum_cardinalities_item_nodes = 0
        for i in range(len(para['item_node_cardinalities'])):
            sum_cardinalities_item_nodes += para['item_node_cardinalities'][i]
        # print(para['item_node_cardinalities'])
        # print(sum_cardinalities_item_nodes)
        # exit(0)
        item_data = torch.FloatTensor(len(item_features[:, 0]), sum_cardinalities_item_nodes).zero_().to(para['device'])
        for j in range(len(para['item_node_cardinalities'])):
            if para['item_node_type'][j] == 'D':
                xx = torch.FloatTensor(len(item_features[:, 0]), para['item_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(item_features[:, j].astype(int)).unsqueeze(1), 1)
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j+1])] = xx
            elif para['item_node_type'][j] == 'C':
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j + 1])] = torch.FloatTensor(item_features[:, j].astype(float)).unsqueeze(1)
        # print(user_data.shape) # torch.Size([358, 118])
        # print(user_data[0])
        # for a in user_data[0]:
        #     print("{:.2f}".format(a.item()) + "  ", end='')
        # print()
        #
        # print(item_data.shape) # torch.Size([358, 88])
        # print(item_data[0])
        # for a in item_data[0]:
        #     print("{:.2f}".format(a.item()) + "  ", end='')
        # print()
        # exit(0)
        features = [user_data, item_data]

    elif para['data_file'] == 'ml-1m':
        print(features[0])
        user_features = features[0][:, 1:features[0].shape[1]]
        item_features = features[1][:, 1:features[1].shape[1]]
        # print(user_features)
        # print(user_features.shape) # (6040, 3)
        # print(item_features)
        # print(item_features.shape) # (3645, 4)
        # exit(0)

        # sum_cardinalities_user_nodes = 0
        # for i in range(BNML.num_nodes):
        #     if BNML.list_node_label[i] == 'UA':
        #         sum_cardinalities_user_nodes += BNML.list_cardinalities[i]
        # user_data = torch.FloatTensor(len(user_features[:, 0]), sum_cardinalities_user_nodes).zero_().to(para['device'])
        # for j in range(BNML.num_nodes):
        #     if BNML.list_node_label[j] == 'UA':
        #         if BNML.list_node_type[j] == 'C':
        #             user_data[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = user_features[:, j].unsqueeze(1)
        #         elif BNML.list_node_type[j] == 'D':
        #             xx = torch.FloatTensor(len(user_features[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, torch.LongTensor(user_features[:, j]).unsqueeze(1)-1, 1)
        #             user_data[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx

        sum_cardinalities_user_nodes = 0
        for i in range(len(para['user_node_cardinalities'])):
            sum_cardinalities_user_nodes += para['user_node_cardinalities'][i]
        # print(sum_cardinalities_user_nodes)
        # exit(0)
        user_data = torch.FloatTensor(len(user_features[:, 0]), sum_cardinalities_user_nodes).zero_().to(para['device'])
        for j in range(len(para['user_node_cardinalities'])):
            if para['user_node_type'][j] == 'C':
                user_data[:, sum(para['user_node_cardinalities'][:j]):sum(para['user_node_cardinalities'][:j+1])] = torch.FloatTensor(user_features[:, j]).unsqueeze(1)
            elif para['user_node_type'][j] == 'D':
                # print(user_features[:, j])
                # exit(0)
                xx = torch.FloatTensor(len(user_features[:, 0]), para['user_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(user_features[:, j].astype(int)).unsqueeze(1)-1, 1)
                # print(xx)
                # exit(0)
                user_data[:, sum(para['user_node_cardinalities'][:j]):sum(para['user_node_cardinalities'][:j+1])] = xx
                # print(user_data)
                # exit(0)
            elif para['user_node_type'][j] == 'D-4-onehot':
                d = torch.LongTensor(user_features[:, j].astype(int))
                # print(d)
                d1 = d % 10
                d2 = (d - d1) % 100 / 10
                d3 = (d - d1 - 10 * d2) % 1000 / 100
                d4 = (d - d1 - 10 * d2 - 100 * d3) % 10000 / 1000
                # print(d1[3644])
                # print(d2[3644])
                # print(d3[3644])
                # print(d4[3644])
                xx1 = torch.FloatTensor(len(user_features[:, 0]), 10).zero_().scatter_(1, d1.long().unsqueeze(1), 1)
                xx2 = torch.FloatTensor(len(user_features[:, 0]), 10).zero_().scatter_(1, d2.long().unsqueeze(1), 1)
                xx3 = torch.FloatTensor(len(user_features[:, 0]), 10).zero_().scatter_(1, d3.long().unsqueeze(1), 1)
                xx4 = torch.FloatTensor(len(user_features[:, 0]), 10).zero_().scatter_(1, d4.long().unsqueeze(1), 1)
                # print(xx4[3644])
                # print(xx3[3644])
                # print(xx2[3644])
                # print(xx1[3644])
                # print(user_data[0])
                user_data[:, 30:40] = xx4
                user_data[:, 40:50] = xx3
                user_data[:, 50:60] = xx2
                user_data[:, 60:70] = xx1
                # print(user_data[0])
                # exit(0)

        sum_cardinalities_item_nodes = 0
        for i in range(len(para['item_node_cardinalities'])):
            sum_cardinalities_item_nodes += para['item_node_cardinalities'][i]
        item_data = torch.FloatTensor(len(item_features[:, 0]), sum_cardinalities_item_nodes).zero_().to(para['device'])
        for j in range(len(para['item_node_cardinalities'])):
            if para['item_node_type'][j] == 'C':
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j+1])] = torch.FloatTensor(item_features[:, j].astype(float)).unsqueeze(1)
            elif para['item_node_type'][j] == 'D':
                xx = torch.FloatTensor(len(item_features[:, 0]), para['item_node_cardinalities'][j]).zero_().scatter_(1, torch.LongTensor(item_features[:, j].astype(int)).unsqueeze(1)-1, 1)
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j+1])] = xx
            elif para['item_node_type'][j] == 'D-k':
                xx = torch.FloatTensor(len(item_features[:, 0]), para['item_node_cardinalities'][j]).zero_()
                for k in range(len(item_features[:, j])):
                    genres = item_features[:, j][k].split("|")
                    for l in genres:
                        xx[k][int(l)-1] = 1.0
                item_data[:, sum(para['item_node_cardinalities'][:j]):sum(para['item_node_cardinalities'][:j+1])] = xx
            elif para['item_node_type'][j] == 'D-4-onehot':
                d = torch.LongTensor(item_features[:, j].astype(int))
                # print(d)
                d1 = d % 10
                d2 = (d - d1) % 100 / 10
                d3 = (d - d1 - 10 * d2) % 1000 / 100
                d4 = (d - d1 - 10 * d2 - 100 * d3) % 10000 / 1000
                # print(d1[3644])
                # print(d2[3644])
                # print(d3[3644])
                # print(d4[3644])
                xx1 = torch.FloatTensor(len(item_features[:, 0]), 10).zero_().scatter_(1, d1.long().unsqueeze(1), 1)
                xx2 = torch.FloatTensor(len(item_features[:, 0]), 10).zero_().scatter_(1, d2.long().unsqueeze(1), 1)
                xx3 = torch.FloatTensor(len(item_features[:, 0]), 10).zero_().scatter_(1, d3.long().unsqueeze(1), 1)
                xx4 = torch.FloatTensor(len(item_features[:, 0]), 10).zero_().scatter_(1, d4.long().unsqueeze(1), 1)
                # print(xx4[3644])
                # print(xx3[3644])
                # print(xx2[3644])
                # print(xx1[3644])
                # print(item_data[0])
                item_data[:, 79:89] = xx4
                item_data[:, 89:99] = xx3
                item_data[:, 99:109] = xx2
                item_data[:, 109:119] = xx1
                # print(item_data[0])
                # print(d[3644])
                # print(item_data[3644])
                # exit(0)

        # print(item_data.shape)
        # print(item_data[3644])
        # for a in item_data[0]:
        #     print("{:.2f}".format(a.item()) + "  ", end='')
        # print()
        # exit(0)
        features = [user_data, item_data]
    return features

def load_BN_trainset(para, test_batch_size, shuffle = True, data_file_path = None, train_loader = None):
    if data_file_path != None:
        train_loader = load_data(para, 'train', data_file_path)
    trainLen = int(para['train'] * len(train_loader))

    testLen = len(train_loader) - trainLen
    if testLen != 0:
        train, test = random_split(train_loader, [trainLen, len(train_loader) - trainLen])
    else:
        train = train_loader
        test = None
    train_dl = torch.utils.data.DataLoader(train, batch_size=para['batch_size'], shuffle=shuffle, pin_memory=True)

    try:
        if test_batch_size == None:
            test_dl = torch.utils.data.DataLoader(test, batch_size=testLen, shuffle=shuffle, pin_memory=True)
        else:
            test_dl = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=shuffle, pin_memory=True)
    except:
        test_dl = None

    # train_dl = torch.utils.data.DataLoader(train_loader, batch_size=para['batch_size'], shuffle=True, pin_memory=True)
    # test_dl = None
    # testLen = 0
    return train_dl, test_dl, trainLen, testLen

# for batch_idx, data in enumerate(tarin_dl): # batch_idx: 0~n
def load_tarin_dl(data, para, BNML, features):
    index = 0 #x 有连续值时必须有
    x = torch.FloatTensor(len(data[:, 0]), BNML.sum_cardinalities).zero_()
    # print(BNML.list_cardinalities)
    # print(BNML.sum_cardinalities)
    # exit(0)
    if para['data_file'] == 'ml-1m':
        U_index = data[:, 0] - 1
        I_index = data[:, 1] - 1
        R = data[:, 2]
        U_features = features[0][U_index]
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'UA':
                x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = U_features[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])]
            elif BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1) - 1, 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        # print(x)
        # print(x.shape)
        # exit(0)
        return x.to(para['device']), U_index.to(para['device']), I_index.to(para['device']), R.to(para['device'])

    elif para['data_file'] == 'toy1' or para['data_file'] == 'toy2' or para['data_file'] == 'child' or para['data_file'] == 'pigs' or para['data_file'] == 'water' or para['data_file'] == 'munin1':
        index = data[:, 0]
        data = data[:, 1:]
        # print(data)
        # print(index)
        # print(features)
        # exit(0)
        i = 0
        j = 0
        while i < len(BNML.list_node_label) and j < len(para['VQVAE_node_label']):
            if BNML.list_node_label[i] == 'V' and para['VQVAE_node_label'][j] == 'V':
                xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[i]).zero_().scatter_(1, torch.LongTensor(data[:,j]).unsqueeze(1),1)
                x[:, sum(BNML.list_cardinalities[:i]):sum(BNML.list_cardinalities[:i + 1])] = xx
                i += 1
                j += 1
            elif BNML.list_node_label[i] == 'V' and para['VQVAE_node_label'][j] != 'V':
                j += 1
            elif BNML.list_node_label[i] != 'V' and para['VQVAE_node_label'][j] == 'V':
                i += 1
            elif BNML.list_node_label[i] != 'V' and para['VQVAE_node_label'][j] != 'V':
                i += 1
                j += 1
            # print(i)
            # print(j)
            # print()
        # print(x)
        # print(x.shape)
        # exit(0)
        return x.to(para['device']), index.to(para['device']), None, None


    elif para['data_file'] == 'dermatology':
        # print(features)
        # exit(0)
        # print(data) # torch.Size([250, 36])
        # print(data.shape) # torch.Size([250, 36])
        # exit(0)
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, 35]
        # print(U_index) # torch.Size([250, 36])
        # print(I_index) # torch.Size([250, 36])
        # print(R) # torch.Size([250, 36])
        # exit(0)
        # U_features = features[0][U_index]
        # print(U_features[0]) # torch.Size([250, 36])
        # print(U_features.shape) # torch.Size([250, 36])
        # exit(0)
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        # print(x)
        # print(x.shape)
        # print(x[0])
        # print(x[3])
        # exit(0)
        return x.to(para['device']), U_index.to(para['device']), I_index.to(para['device']), R.to(para['device'])

    elif para['data_file'] == 'dermatology-origin':
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, 35]
        data = data[:, 1:36]
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
            else:
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                    index += BNML.list_cardinalities[j]
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
                    index += 1
                    # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), U_index, I_index, R.to(para['device'])

    elif para['data_file'] == 'dermatology-NPN-BNML-D':
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, para['num_nodes']]
        data = data[:, 1:(para['num_nodes']+1)]
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
            else:
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                    index += BNML.list_cardinalities[j]
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
                    index += 1
                    # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), U_index, I_index, R.to(para['device'])

    elif para['data_file'] == 'dermatology-NPN-BNML-C' or para['data_file'] == 'bone-marrow-NPN-BNML-C':
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, para['num_nodes']]
        data = data[:, 1:(para['num_nodes']+1)]
        print(U_index)
        print(R[0])
        print(data[0])
        exit(0)
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
            else:
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                    index += BNML.list_cardinalities[j]
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
                    index += 1
                    # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), U_index, I_index, R.to(para['device'])

    elif para['data_file'] == 'dermatology-NPN':
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, 13]
        data = data[:, 1:14]
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
            else:
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                    index += BNML.list_cardinalities[j]
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
                    index += 1
                    # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), U_index, I_index, R.to(para['device'])

    elif para['data_file'] == 'bone-marrow-NPN':
        U_index = data[:, 0] - 1
        I_index = data[:, 0] - 1
        R = data[:, 30]
        data = data[:, 1:31]
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
            else:
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                    index += BNML.list_cardinalities[j]
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
                    index += 1
                    # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), U_index, I_index, R.to(para['device'])

    elif para['data_file'] == 'bone-marrow':
        # print(features)
        # exit(0)
        # print(data) # torch.Size([250, 36])
        # print(data.shape) # torch.Size([250, 36])
        # exit(0)
        U_index = (data[:, 0].long()) - 1
        I_index = (data[:, 0].long()) - 1
        R = data[:, 30]
        # print(U_index) # torch.Size([250, 36])
        # print(I_index) # torch.Size([250, 36])
        # print(R) # torch.Size([250, 36])
        # exit(0)
        # U_features = features[0][U_index]
        # print(U_features[0]) # torch.Size([250, 36])
        # print(U_features.shape) # torch.Size([250, 36])
        # exit(0)
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_label[j] == 'R':
                if BNML.list_node_type[j] == 'C':
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = R.unsqueeze(1)
                elif BNML.list_node_type[j] == 'D':
                    xx = torch.FloatTensor(len(data[:, 0]), BNML.list_cardinalities[j]).zero_().scatter_(1, R.to(int).unsqueeze(1), 1)
                    x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        # print(x)
        # print(x.shape)
        # print(x[0])
        # print(x[3])
        # exit(0)
        return x.to(para['device']), U_index.to(para['device']), I_index.to(para['device']), R.to(para['device'])



    else:
        for j in range(BNML.num_nodes): # 0 ~ num_nodes-1
            if BNML.list_node_type[j] == 'C':
                x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = data[:, index:index+BNML.list_cardinalities[j]]
                index += BNML.list_cardinalities[j]
            elif BNML.list_node_type[j] == 'D':
                xx = torch.FloatTensor(len(data[:,0]), BNML.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1)-1, 1)
                index += 1
                # print((data[:, j].to(int).unsqueeze(1)-1).shape) # torch.Size([30000, 1])
                x[:, sum(BNML.list_cardinalities[:j]):sum(BNML.list_cardinalities[:j+1])] = xx
        return x.to(para['device']), None, None, None