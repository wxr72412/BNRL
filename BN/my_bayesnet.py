import pandas as pd
import itertools
import copy
from copy import deepcopy
import numpy as np
import random
from BN.my_BIF import BIFReader, BIFWriter
from BN import BNSampling
import matplotlib.pyplot as plt
import networkx as nx


def read_excel_data(excelfile):
    bndata_df = pd.DataFrame(pd.read_excel(excelfile, index_col=0))
    return bndata_df


def save_excel(df_all, file_path):
    writer_excel = pd.ExcelWriter(file_path)
    for key, value in df_all.items():
        value.to_excel(writer_excel, sheet_name=key)
    writer_excel.save()


def netSize(variables):
    return len(variables)


def get_index(variables, var):
    try:
        return variables.index(var)
    except:
        print("No node of the name: " + var)
        return None


# Initialise parameters, df is the dataset of samples, net is the constructed BN
def my_init_param(df, net, mis_var):  # apart from initializing the cpt, also convert the cpt to df of cpt
    """
    output:
     var:'age':    age  location    p  count
            0       0         0    0.3   128
            1       1         0    0.7   306
            2       0         1    0.6   209
            3       1         1    0.4   141
            4       0         2    0.9   193
            5       1         2    0.1   23
    """
    # print(df)
    # exit(0)

    for var in net['V']:
        # print(net['cpds'][var])
        # if var == "Disease":
        #     exit(0)
        pares = net['parents'][var]
        # print()
        # print('var:', var)
        if not pares:  # if the pares of var is Null, we only count the num of var in data
            # print(df.groupby(var))
            counts = df.groupby(var).size()
            # print(counts)
            # print(net['learn_cpds_df'][var])
            net['learn_cpds_df'][var]['count'] = counts
            # print(net['learn_cpds_df'][var])
            if var in mis_var:
                net['learn_cpds_df'][var]['count'] = 0
            # print(net['learn_cpds_df'][var])
            # exit(0)
        else:  # we count the num of var+pares in data
            var_pares = [var] + pares
            # print('var_pares:\n', var_pares)
            counts = df.groupby(var_pares).size()  # count the number of var_pares
            # print(counts)
            # exit(0)
            """
            counts(np.series): counts = np.series((0,0,127),(0,1,207),...) 
                            age  location
                0.0  0           127
                     1           207
                     2           193
                1.0  0           305
                     1           140
                     2            23
            """
            bn_counts = pd.DataFrame(counts.index.tolist(), columns=var_pares)  # convert counts into df
            # print(bn_counts)
            # exit(0)
            """
            counts[age,location].index.tolist = [[0,0],[0,1],[0,2],...]
            bn_counts:
            age  location
            0     0
            0     1
            0     2
            1     0
            ....
            """
            # print('counts.tolist:\n', counts.tolist())
            bn_counts['count'] = counts.tolist()
            if var in mis_var:
                bn_counts['count'] = 0
            # print(bn_counts['count'])
            # exit(0)
            # merge the cpt of var and counts of var
            """
            >>net['learn_cpds_df'][var]     >>bn_counts                >>pd.merge
            age  location   p               age  location   count       age  location   p   count
            0     0         0.1             0     0         10          0     0         0.1   10
            0     1         0.7             0     1         20          0     1         0.7   20
            0     2         0.2             0     2         30          0     2         0.2   30
            1     0         0.2             1     0         10          1     0         0.2   10
            ...
            """
            net['learn_cpds_df'][var] = pd.merge(net['learn_cpds_df'][var], bn_counts, on=var_pares, how='outer')
            # print(net['learn_cpds_df'][var])
            # exit(0)

    for var in net['V']:
        normalise_cpt(net, var)
        # print(net['learn_cpds_df'][var])
        # if var == "Disease":
        #     exit(0)
    # exit(0)

    return net


def normalise_cpt(net, var):
    # print('normalize cpt is starting.......')
    # the index of column of cpt_data: ['"x"', 'parents', 'counts', 'p']
    """
        X: "PAP"
        cpt_data generated from init_params:
            "PAP"  "PulmEmbolus"         p  count
        0      0              0         -1       1
        1      1              0         -1      22
        2      2              0         -1      91
        3      0              1         -1     523
        4      1              1         -1    9336
        5      2              1         -1     496
        no_grps: 2
        normalization of cpt_data:
            "PAP"  "PulmEmbolus"         p      count
        0      0              0     0.008772       1
        1      1              0     0.192982      22
        2      2              0     0.798246      91
        3      0              1     0.050507     523
        4      1              1     0.901593    9336
        5      2              1     0.047900     496
    """
    # print(30 * '***')
    # print('var:', var)
    pares = net['parents'][var]
    # print()
    # print(pares)
    cpt = net['learn_cpds_df'][var]  # cpt_data is generated by def.init_params in utils.py
    # print('cpt:\n', cpt)
    cpt['count'] = np.round(cpt['count'], 4)
    # print('count:\n', cpt['count'])
    # print()
    nvals = net['cardinality'][var]  # the number of the possible values of variable
    # print('nvals:', nvals)
    # int(state_combines / nvals)  # the number of groups under different value of current variable
    if not pares:
        no_grps = 1
    else:
        no_grps = len(net['cpds'][var])  # net['cpds'] different from net['cpds_df']
    # print('no_grps:', no_grps)
    df_new = pd.DataFrame()
    i = 0
    for n in range(no_grps):
        curr_df = cpt.iloc[i:i + nvals, :].copy()  # df appended to df needs to use 'copy()', otherwise, report errors
        # print(curr_df)
        curr_df_copy = curr_df.loc[:, 'count'].copy()
        # print('curr_df_copy:\n', curr_df_copy)
        mis_index_count = np.argwhere(np.isnan(np.asarray(curr_df_copy)))  # [[2], [3], [5]]
        # print('mis_index_count:', mis_index_count)
        if mis_index_count.size == 0:
            curr_df['p'] = normalise_counts_rand(curr_df_copy)  # normalize counts and reassign the value to 'p'
            # print(curr_df['p'])
            # else:
            #     for j in mis_index_count:
            #         # curr_df_copy.iloc[j[0]] = 0.0001
            #         curr_df_copy.iloc[j[0]] = round(random.uniform(0.01, 0.001), 3)
            curr_df.loc[:, 'p'] = normalise_counts_rand(curr_df_copy)
            # print(curr_df.loc[:, 'p'])
            # curr_df.loc[:, 'count'] = curr_df_copy
        df_new = df_new.append(curr_df)
        # print(df_new)
        i = i + nvals
    net['learn_cpds_df'][var] = df_new


# normalise a pd.Series of counts such as curr_df['counts']={0:12,1:11,2:1270}
def normalise_counts_rand(count_series):  # counts represents curr_df['counts'], curr_df is df.series
    normalised_vals = []
    denom = np.sum(count_series)
    if denom == 0:
        count_series += 0.05
        denom = np.sum(count_series)
    for val in count_series:
        cal_result = round(val / float(denom), 3)
        normalised_vals.append(cal_result)
        # print('normalised_vals:', normalised_vals)
    return normalised_vals


def PMI_construction(net, nodes, edges, var_card, size):
    """
            nodes_state_count = {'amenities': {0: 0, 1: 0}, 'neighborhood': {0: 0, 1: 0},...}
    """
    nodes_state_count = {node: {node_state: 0 for node_state in range(var_card[node])} for node in nodes}

    edges_state_count = {edge: {state_combination: 0 for state_combination in
                                itertools.product(*[range(var_card[node])
                                                    for node in edge])} for edge in edges}
    """
    edges_state_values:
        {('amenities', 'location'): {(0, 0): 0,  (0, 1): 0,  (0, 2): 0, (1, 0): 0, (1, 1): 0, (1, 2): 0},...}
    """
    edges_state_values = copy.deepcopy(edges_state_count)  # copy.deepcopy
    # edges_state matrix is initialized
    e_column_index = set()  # the key idea of set is non-repetitive
    """
        e_column_index = {(0, 0), (0, 1), (0, 2), ...}
    """
    for edge in edges:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            e_column_index.add(state_combination)
    e_column_index = list(e_column_index)
    e_column_index.sort()

    """ 
    E_ijk:
                                  (0, 0)  (0, 1)  (0, 2)  ...  (2, 0)  (2, 1)  (2, 2)
        (amenities, location)          0       0       0  ...       0       0       0
        (neighborhood, location)       0       0       0  ...       0       0       0
        (neighborhood, children)       0       0       0  ...       0       0       0
        (location, age)                0       0       0  ...       0       0       0
        (children, schools)            0       0       0  ...       0       0       0
        (location, price)              0       0       0  ...       0       0       0
        (age, price)                   0       0       0  ...       0       0       0
        (schools, price)               0       0       0  ...       0       0       0
    """
    samples_Mynew = BNSampling.Mynew_forward_sampling(net, nodes).My_forward_sample(size, return_type='dataframe')
    """
     nodes_state_count: calculate #(X_i)和#(X_j)
       {'size': {0: 326, 1: 340, 2: 334}, 'neighborhood': {0: 407, 1: 593}, ...}
    """
    for node in nodes:
        for i in range(len(samples_Mynew[node])):
            nodes_state_count[node][samples_Mynew[node][i]] += 1
    # print('nodes_state_count:\n',nodes_state_count)
    """
    edge states of graph are counted: calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
    """
    for edge in edges:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            # print('edge：{edge} state:{state_combination}'.format(edge=edge, state_combination=state_combination))
            for i in range(len(samples_Mynew)):
                if samples_Mynew.loc[i, edge[0]] == state_combination[0] \
                        and samples_Mynew.loc[i, edge[1]] == state_combination[1]:
                    edges_state_count[edge][state_combination] += 1
            # calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
            if nodes_state_count[edge[0]][state_combination[0]] != 0 and \
                    nodes_state_count[edge[1]][state_combination[1]] != 0:
                edges_state_values[edge][state_combination] = round((edges_state_count[edge][state_combination] * size)
                                                                    / (nodes_state_count[edge[0]][
                                                                           state_combination[0]] *
                                                                       nodes_state_count[edge[1]][
                                                                           state_combination[1]]), 3)
            else:
                edges_state_values[edge][state_combination] = 0

    # generate the index of node, suach as: [X_1,X_2,X_3,...,X_n]
    n_row_index = []
    for i in range(len(nodes)):
        for j in range(var_card[nodes[i]]):
            n_row_index.append('{u}_{s}'.format(u=nodes[i], s=j))
            # print('\n',n_row_index)

    # initialize node_state matirx
    N_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    # print(N_ijk)
    PMI_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    # print('PMI_ijk:\n',PMI_ijk)

    # construct PMI matirx logN_ijk-logk
    # first construct the nodes_state matrix, make the edges_state matrix transformed to nodes_state matrix
    k = 1
    e_weight = []
    for edge in edges:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            # print('state_combination:',state_combination)
            edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
            edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
            #         print(edge_0,edge_1)
            N_ijk.loc[edge_0, edge_1] = edges_state_values[edge][state_combination]
            # PMI=max(PMI,0)
            if N_ijk.loc[edge_0, edge_1] == 0:
                PMI_ijk.loc[edge_0, edge_1] = 0
            else:
                PMI_ijk.loc[edge_0, edge_1] = round(np.log(N_ijk.loc[edge_0, edge_1]) - np.log(k), 3)
                if PMI_ijk.loc[edge_0, edge_1] < 0:
                    PMI_ijk.loc[edge_0, edge_1] = 0
                if PMI_ijk.loc[edge_0, edge_1] > 0:
                    e_weight.append((edge_0, edge_1, PMI_ijk.loc[edge_0, edge_1]))
    return PMI_ijk, e_weight


# use for DEBN's PMI updating
def update_PMI_construction(net, nodes, edges, var_card, mis_index, size, pre_PMI_IJK, edge_weight):
    """
            nodes_state_count = {'amenities': {0: 0, 1: 0}, 'neighborhood': {0: 0, 1: 0},...}
    """
    mis_var_store = [net['V'][i] for i in mis_index]
    # print('mis_var_store:', mis_var_store)
    # edge_store stores the edges between mis_var and its children
    mis_var_edge_store = [edge for edge in edges for mis_var in mis_var_store if mis_var == edge[0]]
    mis_var_children_store = list(set(node for edge in mis_var_edge_store for node in edge))
    # mis_var_children_card = {node: card for node, card in net['cardinality'].items()
    # if node in mis_var_children_store}
    # print('mis_var_edge_store:', mis_var_edge_store)

    mis_var_state_count = {node: {node_state: 0 for node_state in range(var_card[node])}
                           for node in mis_var_children_store}

    mis_var_edges_state_count = {edge: {state_combination: 0 for state_combination in
                                        itertools.product(*[range(var_card[node])
                                                            for node in edge])} for edge in mis_var_edge_store}
    """
    edges_state_values:
        {('amenities', 'location'): {(0, 0): 0,  (0, 1): 0,  (0, 2): 0, (1, 0): 0, (1, 1): 0, (1, 2): 0},...}
    """
    edges_state_values = copy.deepcopy(mis_var_edges_state_count)  # copy.deepcopy

    samples_Mynew = BNSampling.Mynew_forward_sampling(net, nodes).My_forward_sample(size, return_type='dataframe')
    """
     nodes_state_count: calculate #(X_i)和#(X_j)
       {'size': {0: 326, 1: 340, 2: 334}, 'neighborhood': {0: 407, 1: 593}, ...}
    """
    for node in mis_var_children_store:
        for i in range(len(samples_Mynew[node])):
            mis_var_state_count[node][samples_Mynew[node][i]] += 1
    # print('nodes_state_count:\n',nodes_state_count)
    """
    edge states of graph are counted: calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
    """
    for edge in mis_var_edge_store:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            # print('edge：{edge} state:{state_combination}'.format(edge=edge, state_combination=state_combination))
            for i in range(len(samples_Mynew)):
                if samples_Mynew.loc[i, edge[0]] == state_combination[0] \
                        and samples_Mynew.loc[i, edge[1]] == state_combination[1]:
                    mis_var_edges_state_count[edge][state_combination] += 1
            # calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
            if mis_var_state_count[edge[0]][state_combination[0]] != 0 and \
                    mis_var_state_count[edge[1]][state_combination[1]] != 0:
                edges_state_values[edge][state_combination] = round(
                    (mis_var_edges_state_count[edge][state_combination] * size)
                    / (mis_var_state_count[edge[0]][state_combination[0]] *
                       mis_var_state_count[edge[1]][state_combination[1]]), 3)
            else:
                edges_state_values[edge][state_combination] = 0

    # generate the index of mis_var and its children, such as: [X_1,X_2,X_3,...,X_n]
    n_row_index = []
    for i in range(len(mis_var_children_store)):
        for j in range(var_card[mis_var_children_store[i]]):
            n_row_index.append('{u}_{s}'.format(u=mis_var_children_store[i], s=j))
            # print('\n',n_row_index)

    # initialize node_state matirx
    N_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    # print(N_ijk)

    # construct PMI matirx logN_ijk-logk
    # first construct the nodes_state matrix, make the edges_state matrix transformed to nodes_state matrix
    k = 1
    for edge in mis_var_edge_store:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            #         print('state_combination:',state_combination)
            edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
            edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
            #         print(edge_0,edge_1)
            N_ijk.loc[edge_0, edge_1] = edges_state_values[edge][state_combination]
            # PMI=max(PMI,0)
            if N_ijk.loc[edge_0, edge_1] == 0:
                pre_PMI_IJK.loc[edge_0, edge_1] = 0
            else:
                pre_PMI_IJK.loc[edge_0, edge_1] = round(np.log(N_ijk.loc[edge_0, edge_1]) - np.log(k), 3)
                if pre_PMI_IJK.loc[edge_0, edge_1] < 0:
                    pre_PMI_IJK.loc[edge_0, edge_1] = 0
                if pre_PMI_IJK.loc[edge_0, edge_1] > 0:
                    for e_weight in edge_weight:
                        if e_weight[0] == edge_0 and e_weight[1] == edge_1:
                            edge_weight.remove(e_weight)
                            edge_weight.append((edge_0, edge_1, pre_PMI_IJK.loc[edge_0, edge_1]))

    return pre_PMI_IJK, edge_weight


# use for TRIP's delta_PMI
def delta_PMI_construction(net, nodes, edges, var_card, mis_index, size, PMI_IJK_index_name):
    """
            nodes_state_count = {'amenities': {0: 0, 1: 0}, 'neighborhood': {0: 0, 1: 0},...}
    """
    mis_var_store = [net['V'][i] for i in mis_index]
    # print('mis_var_store:', mis_var_store)
    # edge_store stores the edges between mis_var and its children
    mis_var_edge_store = [edge for edge in edges for mis_var in mis_var_store if mis_var == edge[0]]
    mis_var_children_store = list(set(node for edge in mis_var_edge_store for node in edge))
    # if node in mis_var_children_store}
    # print('mis_var_edge_store:', mis_var_edge_store)

    mis_var_state_count = {node: {node_state: 0 for node_state in range(var_card[node])}
                           for node in mis_var_children_store}

    mis_var_edges_state_count = {edge: {state_combination: 0 for state_combination in
                                        itertools.product(*[range(var_card[node])
                                                            for node in edge])} for edge in mis_var_edge_store}
    """
    edges_state_values:
        {('amenities', 'location'): {(0, 0): 0,  (0, 1): 0,  (0, 2): 0, (1, 0): 0, (1, 1): 0, (1, 2): 0},...}
    """
    edges_state_values = copy.deepcopy(mis_var_edges_state_count)  # copy.deepcopy

    samples_Mynew = BNSampling.Mynew_forward_sampling(net, nodes).My_forward_sample(size, return_type='dataframe')
    """
     nodes_state_count: calculate #(X_i)和#(X_j)
       {'size': {0: 326, 1: 340, 2: 334}, 'neighborhood': {0: 407, 1: 593}, ...}
    """
    for node in mis_var_children_store:
        for i in range(len(samples_Mynew[node])):
            mis_var_state_count[node][samples_Mynew[node][i]] += 1
    # print('nodes_state_count:\n',nodes_state_count)
    """
    edge states of graph are counted: calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
    """
    for edge in mis_var_edge_store:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            # print('edge：{edge} state:{state_combination}'.format(edge=edge,state_combination=state_combination))
            for i in range(len(samples_Mynew)):
                if samples_Mynew.loc[i, edge[0]] == state_combination[0] \
                        and samples_Mynew.loc[i, edge[1]] == state_combination[1]:
                    mis_var_edges_state_count[edge][state_combination] += 1
            # calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
            if mis_var_state_count[edge[0]][state_combination[0]] != 0 \
                    and mis_var_state_count[edge[0]][state_combination[0]] != 0:
                edges_state_values[edge][state_combination] = round(
                    (mis_var_edges_state_count[edge][state_combination] * size)
                    / (mis_var_state_count[edge[0]][state_combination[0]] *
                       mis_var_state_count[edge[0]][state_combination[0]]), 3)
            else:
                edges_state_values[edge][state_combination] = 0

    # generate the index of mis_var and its children, such as: [X_1,X_2,X_3,...,X_n]
    n_row_index = []
    for i in range(len(mis_var_children_store)):
        for j in range(var_card[mis_var_children_store[i]]):
            n_row_index.append('{u}_{s}'.format(u=mis_var_children_store[i], s=j))
            # print('\n',n_row_index)

    # initialize node_state matirx
    N_ijk = pd.DataFrame(0, index=PMI_IJK_index_name, columns=PMI_IJK_index_name)
    # print(N_ijk)

    # construct PMI matirx logN_ijk-logk
    # first construct the nodes_state matrix, make the edges_state matrix transformed to nodes_state matrix
    k = 1
    for edge in mis_var_edge_store:
        for state_combination in itertools.product(*[range(var_card[node]) for node in edge]):
            #         print('state_combination:',state_combination)
            edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
            edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
            #         print(edge_0,edge_1)
            N_ijk.loc[edge_0, edge_1] = edges_state_values[edge][state_combination]
            # PMI=max(PMI,0)
            # N_ijk.loc[edge_0, edge_1] = round(np.log(N_ijk.loc[edge_0, edge_1]) - np.log(k), 3)
            if N_ijk.loc[edge_0, edge_1] != 0:
                N_ijk.loc[edge_0, edge_1] = round(np.log(N_ijk.loc[edge_0, edge_1]) - np.log(k), 3)
            if N_ijk.loc[edge_0, edge_1] < 0:
                N_ijk.loc[edge_0, edge_1] = 0

    return N_ijk  # edge_weight

def retrieve_p(bn, mis_var, X_pares_value_store, mis_var_value_store):
    """
    input:
        bn, mis_var,
        X_pares_value_store: {'pare0':0, 'pare0':1, 'pare1':0, ...}
        Sim_mis_var: Sim(x|neighbor(x))=Sim(pa(x)->X)+Sim(x->child(x)), e.g. [-1.62525305 -0.23173922]
    output:
        normalised_vals.append(val / float(denom))

    """
    # print('mis_var: {k} {v}'.format(k=mis_var, v=mis_var_value_store[mis_var]))

    mis_var_pare_df = bn['learn_cpds_df'][mis_var]
    """
    X_pares_value_store such as {'pare0':0, 'pare0':1, 'pare1':0, ...}
    """
    # print('X_pares_value_store:', X_pares_value_store)
    # if the mis_var has no parents
    if len(X_pares_value_store) == 0:
        temp_value = mis_var_pare_df['p'][mis_var_value_store[mis_var]]

    else:  # if the mis_var has  parents
        """
        var = {'Rank': 0}, pares = {'Year': 1, 'Points': 1}
        dict(var, **pares) = {'Rank': 0, 'Year': 1, 'Points': 1}
        """
        var_pares = dict(mis_var_value_store, **X_pares_value_store)
        var_pares_name = []
        var_pares_value = []
        for i, j in var_pares.items():
            var_pares_name.append(i)
            var_pares_value.append(j)
        # [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), ...]
        pare_var_index = mis_var_pare_df.set_index(var_pares_name).index.tolist()
        index = pare_var_index.index(tuple(var_pares_value))  # the index in [(0, 0, 0), (0, 0, 1),  ...]
        temp_value = mis_var_pare_df.loc[index, 'p']
    return temp_value

def retrieve_pare_p(bn, mis_var, X_pares_value_store):
    """
    input:
        bn, mis_var,
        X_pares_value_store: {'pare0':0, 'pare0':1, 'pare1':0, ...}
        Sim_mis_var: Sim(x|neighbor(x))=Sim(pa(x)->X)+Sim(x->child(x)), e.g. [-1.62525305 -0.23173922]
    output:
        normalised_vals.append(val / float(denom))

    """
    # print('mis_var: {k} {v}'.format(k=mis_var, v=mis_var_value_store[mis_var]))

    mis_var_pare_df = bn['learn_cpds_df'][mis_var]
    """
    X_pares_value_store such as {'pare0':0, 'pare0':1, 'pare1':0, ...}
    """
    # print('X_pares_value_store:', X_pares_value_store)
    # if the mis_var has no parents
    if len(X_pares_value_store) == 0:
        temp_value = mis_var_pare_df

    else:  # if the mis_var has  parents
        """
        var = {'Rank': 0}, pares = {'Year': 1, 'Points': 1}
        dict(var, **pares) = {'Rank': 0, 'Year': 1, 'Points': 1}
        """
        for i, j in X_pares_value_store.items():
            pares = i
            pares_value = j

        temp_value = mis_var_pare_df[mis_var_pare_df[pares] == pares_value]
    return temp_value


if __name__ == '__main__':
    # %%
    reader = BIFReader('../data/network.bif')  # Very large BN
    net = reader.my_model()

    # writer = BIFWriter(net)
    # writer.write_bif('../data/network_initparam_written.bif')
    #  nodes = net['V']
    edges = [tuple(edge) for edge in net['E']]
    nodes = BNSampling.topological_nodes(edges)
    """
       edges = [['amenities', 'location'], ['neighborhood', 'location'],...]
    """
    var_card = net['cardinality']
    size = 1000
    PMI_IJK, _ = PMI_construction(net, nodes, edges, var_card, size)
    # %%
    print('net:\n', net)
    df = read_excel_data('../data/bn_data_1000.xlsx')
    print('df:\n', df)
    # %%
    # net_test = my_init_param(df, net)
    # # print('net_test:\n', net_test)
    # # %%
    # reader = BIFReader('../data/network.bif')  # Very large BN
    # net = reader.my_model()
    # edges = [tuple(edge) for edge in net['E']]
    # nodes = BNSampling.topological_nodes(edges)
    # var_card = net['cardinality']
    # mis_index = [1, 6, 7]
    # pre_PMI_IJK = deepcopy(PMI_IJK)
    # size = 1000
    # # PMI_IJK_update = update_PMI_construction(net, nodes, edges, var_card, mis_index, size, pre_PMI_IJK)
    #
    # # %%
    #
    # reader = BIFReader('../data/network.bif')  # Very large BN
    # net = reader.my_model()
    #
    # writer = BIFWriter(net)
    # writer.write_bif('../data/network_initparam_written.bif')
    # #  nodes = net['V']
    # edges = [tuple(edge) for edge in net['E']]
    # nodes = BNSampling.topological_nodes(edges)
    # """
    #    edges = [['amenities', 'location'], ['neighborhood', 'location'],...]
    # """
    # var_card = net['cardinality']
    # size = 1000
    # edge_weight = []
    # PMI_IJK, edge_weight = PMI_construction(net, nodes, edges, var_card, size)
    # G = nx.DiGraph()
    # G.add_nodes_from(list(PMI_IJK.index))
    # G.add_weighted_edges_from(edge_weight)
    #
    # # %%
    # reader = BIFReader('../data/network.bif')  # Very large BN
    # net = reader.my_model()
    #
    # df = read_excel_data('../data/bn_data_10_multivar.xlsx')
    # net = my_init_param(df, net)
