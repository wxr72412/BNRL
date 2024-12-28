import torch


def find_cpt(net, pa, pa_values, V, V_value=None):
    # print(pa)
    # print(pa_values)
    # print(V)
    # print(type(V))
    # print(V_value)
    # print(type(V_value))
    # exit(0)
    df = net['learn_cpds_df'][V]
    r = df
    # print(r)
    if pa != []:
        for p, v in zip(pa, pa_values):
            r = r.loc[(df[p] == v), :]

    if V_value != None:
        return float(r.loc[(df[V] == V_value), :]['p'].values)
    else:
        return [float(r.loc[(df[V] == c), :]['p'].values) for c in range(net['cardinality'][V])]


def change_cpt(net, pa, pa_values, V, V_value, cpt_value):
    df = net['learn_cpds_df'][V]
    for p, v in zip(pa, pa_values):
        df = df.loc[(df[p] == v), :]
    index = df.loc[(df[V] == V_value), :].index.tolist()[0]
    # print(index)

    net['learn_cpds_df'][V].loc[index, 'p'] = cpt_value
    # print(net['learn_cpds_df'][V])



def change_all_cpt(net, BNML, para, curr_iter = None):
    data_m_temp = torch.FloatTensor(1, BNML.sum_cardinalities).zero_().to(para['device'])
    data_s = torch.zeros(data_m_temp.size()).to(para['device'])
    for V_net in net['V']:

        # if int(V_net) > 4:
        #     continue



        df = net['learn_cpds_df'][V_net]
        # print(df)
        # print('V_net: ' + str(V_net))

        num_paras_V_net = df.shape[0]
        # print(num_paras_V_net)
        V_BNML = para['list_node_index_VQVAE'].index(int(V_net))

        # if curr_iter > 1 and BNML.list_dependent_latent_node(V_BNML) == []: # 除了第一次迭代外，其他时候不学习与隐变量无关的显变量的CPD
        # if BNML.list_dependent_latent_node(V_BNML) == []:
        #     continue
        # if V_BNML + 1 != 19:  # CHILD  L=2 最后一个变量
        #     continue


        # print('V_BNML: ' + str(V_BNML))
        # print(BNML.list_parents(V_BNML))
        # print(BNML.list_cardinalities)
        num_paras_V_BNML = BNML.parameters_node_i(V_BNML)
        # print(num_paras_V_BNML)

        parents = list(df.columns)
        parents.remove(V_net)
        parents.remove('p')
        parents.remove('count')
        # print(parents)
        # exit(0)

        if num_paras_V_net != num_paras_V_BNML:
            print('Node: ' + str(V_net) + '. num_paras_V_net is not equal to num_paras_V_BNML')
            print(num_paras_V_net)
            print(num_paras_V_BNML)
            exit(0)
        else:
            for index in range(num_paras_V_net):
                # print(index)
                pa_values = []
                line = df.iloc[index]
                # print(line)
                data_m = data_m_temp.clone().detach()  # 避免修改data_m的时候将data_m_temp修改
                # print(data_m)
                if parents == []:
                    pass
                else:
                    for pa in parents:
                        # print(pa)
                        pa_BNML = para['list_node_index_VQVAE'].index(int(pa))
                        # print(pa_BNML)
                        pa_value = int(line[pa])
                        # print(pa_value)
                        pa_values.append(pa_value)

                        q = torch.ones(1).to(para['device']) * pa_value
                        # print(q)
                        xx = torch.FloatTensor(1, BNML.list_cardinalities[pa_BNML]).to(para['device']).zero_().scatter_(1, q.to(int).unsqueeze(1), 1)
                        # print(xx)
                        data_m[:, sum(BNML.list_cardinalities[:pa_BNML]):sum(BNML.list_cardinalities[:pa_BNML + 1])] = xx
                        # print(data_m)
                        # exit(0)

                # print(data_m)
                output = BNML.list_NPN[V_BNML]((data_m, data_s))[0][0]
                # print(output)

                V_value = int(line[V_net])
                # print(V_value)

                cpt_value = output[V_value].item()
                # print(cpt_value)

                change_cpt(net, parents, pa_values, V_net, V_value, cpt_value)

                # print(net['learn_cpds_df'][V_net])
                # exit(0)
    # exit(0)

#
#
# def change_back_all_cpt(net, BNML, para):
#     data_m_temp = torch.FloatTensor(1, BNML.sum_cardinalities).zero_().to(para['device'])
#     data_s = torch.zeros(data_m_temp.size()).to(para['device'])
#     for V_net in net['V']:
#         df = net['learn_cpds_df'][V_net]
#         # print(df)
#
#
#         num_paras_V_net = df.shape[0]
#         # print(num_paras_V_net)
#         V_BNML = para['list_node_index_VQVAE'].index(int(V_net))
#         # print(V_BNML)
#         num_paras_V_BNML = BNML.parameters_node_i(V_BNML)
#         # print(num_paras_V_BNML)
#
#         parents = list(df.columns)
#         parents.remove(V_net)
#         parents.remove('p')
#         parents.remove('count')
#         # print(parents)
#         # exit(0)
#
#         if num_paras_V_net != num_paras_V_BNML:
#             print('Node: ' + str(V_net) + '. num_paras_V_net is not equal to num_paras_V_BNML')
#             exit(0)
#         else:
#             for index in range(num_paras_V_net):
#                 # print(index)
#                 pa_values = []
#                 line = df.iloc[index]
#                 # print(line)
#                 data_m = data_m_temp.clone().detach()  # 避免修改data_m的时候将data_m_temp修改
#                 # print(data_m)
#                 if parents == []:
#                     pass
#                 else:
#                     for pa in parents:
#                         # print(pa)
#                         pa_BNML = para['list_node_index_VQVAE'].index(int(pa))
#                         # print(pa_BNML)
#                         pa_value = int(line[pa])
#                         # print(pa_value)
#                         pa_values.append(pa_value)
#
#                         q = torch.ones(1).to(para['device']) * pa_value
#                         # print(q)
#                         xx = torch.FloatTensor(1, BNML.list_cardinalities[pa_BNML]).to(para['device']).zero_().scatter_(1, q.to(int).unsqueeze(1), 1)
#                         # print(xx)
#                         data_m[:, sum(BNML.list_cardinalities[:pa_BNML]):sum(BNML.list_cardinalities[:pa_BNML + 1])] = xx
#                         # print(data_m)
#                         # exit(0)
#
#                 # print(data_m)
#                 output = BNML.list_NPN[V_BNML]((data_m, data_s))[0][0]
#                 # print(output)
#
#                 V_value = int(line[V_net])
#                 # print(V_value)
#
#                 cpt_value = output[V_value].item()
#                 # print(cpt_value)
#
#                 change_cpt(net, parents, pa_values, V_net, V_value, cpt_value)
