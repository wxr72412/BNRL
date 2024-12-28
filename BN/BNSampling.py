import itertools
import networkx as nx
import numpy as np
import pandas as pd
import sys

sys.path.append('..')  # return to the previous level to import package
from BN.my_BIF import BIFReader
from pgmpy.utils.mathext import sample_discrete
from pgmpy.sampling import _return_samples


def topological_nodes(eg):  # eg denotes the edges of G
    G = nx.DiGraph()
    G.add_edges_from(eg)
    # G_edges = [e for e in G.edges_iter()]
    # print('G_edges:\n', G_edges)
    topological_node = list(nx.topological_sort(G))
    return topological_node


# 将{'location':'bad'}转化为{'location':1}的形式
def state_to_num(net, node):  # node denotes Query_node = {'children': 'bad'}
    node_num = {}
    for keys, values in list(node.items()):
        nstate = net['states'][list(node.keys())[0]]  # find node[i] state
        nstate_index = nstate.index(values)
        #         print('keys:',keys)
        node_num[keys] = nstate_index
    #         print('node:',node)
    return node_num


# 将probability={0:0.4,1:0.6}转化为{'old':0.4,'new':0.6}的形式
def pnum_to_state(net, node, probabilites):
    for keys, values in list(probabilites.items()):
        nstate = net['states'][list(node.keys())[0]][keys]
        probabilites.update({nstate: probabilites.pop(keys)})
        dict.pop
    return probabilites


def read_excel(excelfile):
    bndata_df = pd.DataFrame(pd.read_excel(excelfile, index_col=0))
    return bndata_df


def save_excel(df_all, file_path):
    writer_excel = pd.ExcelWriter(file_path)
    for key, value in df_all.items():
        value.to_excel(writer_excel, sheet_name=key)
    writer_excel.save()


def sampling_for_EM(bn, latent_vars):
    """
    Input:
        bn: bn model generated by
            reader = BIFReader('../data/network.bif')  # network
            network = reader.my_model()
        latent_vars: the list of latent variables, e.g. latent_vars = ['age', 'location']
    Ootput:
    """
    nodes = bn['V']
    state_combinations = itertools.product(*[range(bn['cardinality'][var]) for var in nodes])
    state_combinations_list = list(state_combinations)
    samples = pd.DataFrame(state_combinations_list, columns=nodes)
    if not latent_vars:
        return samples
    else:
        for i in range(len(latent_vars)):
            samples[latent_vars[i]] = np.nan
        return samples

def sampling_for_allsamples(nodes, node_cards):
    """
    Input:
        nodes:
            nodes = ['size', 'neighborhood', 'children', 'schools']
        node_cards:
            node_cards = {'size': 2, 'neighborhood': 3, 'children': 2, 'schools': 4}
    Ootput:
    """
    # bn = {'size': 2, 'neighborhood': 3, 'children': 2, 'schools': 4}
    # nodes = ['size', 'neighborhood', 'children', 'schools']
    df = pd.DataFrame([], columns=nodes)
    for node in nodes:
        if len(df.index) == 0:
            df.loc[:, node] = list(range(node_cards[node]))
        else:
            if np.isnan(df.loc[0, node]):
                for r in range(df.shape[0]):
                    new_row = pd.DataFrame(df.loc[0, :]).T
                    for n in range(node_cards[node]):
                        new_row.iloc[0, nodes.index(node)] = n
                        df = pd.concat([df, new_row])
                        # print('df:\n', df)
                        df = df.reset_index(drop=True)
                    df.drop([0], axis=0, inplace=True)
                    # print('df:\n', df)
                    df = df.reset_index(drop=True)
                print(50 * '*' + node + 50 * '*')
                print('df:\n', df)
            else:
                break
    return df

class Mynew_forward_sampling:

    def __init__(self, net, topology):
        self.net = net
        self.topological_order = topology

    def My_forward_sample(self, size=1, return_type='dataframe'):
        # topological_order: ['size', 'neighborhood', 'children', 'schools', 'amenities', 'location', 'age', 'price']
        # types=[('size','int'),('neighborhood','int'),...] 共8个节点
        types = [(var_name, 'int') for var_name in self.topological_order]
        # print('types:\n', types)

        """
        sampled=np.zeros(size, dtype=types)
         'size', 'neighborhood', 'children', 'schools', 'amenities', 'location', 'age', 'price'
        1 ( 0           0             0            0          0           0         0       0 )
        2 ( 0           0             0            0          0           0         0       0 )
        ....
        """
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        # print('sampled:\n', sampled)
        for node in self.topological_order:
            # print('node:', node)
            states = range(self.net['cardinality'][node])
            # evidence denotes the parents of variable, cpd.variables denotes the all variables of this cpd
            evidence = self.net['parents'][node]
            # print('evidence:', evidence)
            if evidence:
                """
               cached_values:{(evi_0,evi_1):[query_0,query_1,query_2]}
               cached_values: {(   0,    0): array([0.3, 0.4, 0.3])}
                """
                cached_values = self.My_pre_compute_reduce(variable=node)
                # print('computed_cached_values:\n', cached_values)
                # [print('sampled[{i}]:\n{s}'.format(i=i, s=sampled[i])) for i in evidence]
                # sampled[i], e.g. sampled[neighborhood]=[1 1 0 ... 1 1 1]
                """
                np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组,
                e.g. sampled[neighborhood]=[1 1 0 ... 1 1 1], sampled[amenities]=[1 0 1 ... 1 1 1]
                np.vstack([sampled[neighborhood],sampled[amenities]]) =
                 [[1 1 0 ... 1 1 1]
                 [1 0 1 ... 1 1 1]]
                """
                # sample[evidence] is known because we traverse node according to the topological sort
                evidence = np.vstack([sampled[i] for i in evidence])
                # print('evidence_vstack:\n', evidence)
                # print('vstack_evidence and evidence.T:\n', evidence, '\n', evidence.T)
                """  
                map(function, iterable),fun指函数，iterable指列表，map(lambda x: x ** 2, [1, 2, 3, 4, 5])
                map输出的值为[1,4,9,16.25]
                """
                weights = list(map(lambda t: cached_values[tuple(t)], evidence.T))  # tuple(t)是将列表[0,1]等转换为元组(0,1)
                # print('weights:\n', weights)
            else:  # in terms of topological sort, we first perform root node.
                weights = self.net['cpds'][node]  # cpd.values: [0.33 0.34 0.33] <class 'numpy.ndarray'>
            # print('weights:', weights)
            sampled[node] = sample_discrete(states, weights, size)
            #  print('sampled_node:\n',sampled,type(sampled))

        return _return_samples(return_type, sampled)

    def My_pre_compute_reduce(self, variable):

        """
        The cpd of 'children'
         neighborhood | neighborhood_0 | neighborhood_1 |
        +--------------+----------------+----------------+
        | children_0   | 0.6            | 0.3            |
        +--------------+----------------+----------------+
        | children_1   | 0.4            | 0.7            |
        +--------------+----------------+----------------+
        """
        # variable_cpd = self.model.get_cpds(variable)
        variable_evid = self.net['parents'][variable]  # 从后往前数的话，最后一个位置为-1,[:0:-1]表示从倒数第一位到正数第二位
        # print('variable_evid:\n', variable_evid)
        cpd_values = self.net['cpds'][variable]
        cached_values = {}
        # *[range(self.cardinality[var])指列表list[]里的所有range元素，如[range(1),range(2),range(3)]
        # list(itertools.product(range(2),range(2))) is cartesian product [(0,0),(0,1),(1,0),(1,1)]
        state_combinations = itertools.product(*[range(self.net['cardinality'][var]) for var in variable_evid])
        for state_combi, cpd_value in zip(state_combinations, cpd_values):
            cached_values[state_combi] = cpd_value
            # print('cached_values:', cached_values)
        """
        cached_values: {(0,): array([0.3, 0.7]), (1,): array([0.6, 0.4])}
        """

        return cached_values


if __name__ == '__main__':
    # reader = BIFReader('../data/network.bif')  # network
    # latent_vars = ['age', 'price']

    # reader = BIFReader('../data/bn/child.bif')  # child
    # latent_vars = ['Disease', 'LungParench']

    reader = BIFReader('../data/bn/alarm.bif')  # alarm
    # latent_vars = ['TPR', 'CO']

    # reader = BIFReader('../data/bn/hepar2.bif')  # hepar2
    # latent_vars = ['surgery', 'fat']

    # reader = BIFReader('../data/bn/andes.bif')  # andes
    # latent_vars = ['SNode_3', 'SNode_4']

    # reader = BIFReader('../data/bn/munin1.bif')  # munin1
    # latent_vars = ['R_MYAS_APB_MUDENS', 'R_LNLW_APB_MALOSS']

    # reader = BIFReader('../data/bn/link.bif')  # link
    # latent_vars = ['N73_a_m', 'N73_a_f']

    network = reader.my_model()
    # print('network:\n', network)
    # nodes = network['V']
    # print('nodes:', nodes)
    #
    # print('number of nodes:', len(nodes))
    # edges = network['E']
    # print('edges:\n', edges)
    # print('number of edges:', len(edges))
    # # topological_nodes = topological_nodes(edges)
    # # print('topological_nodes:\n', topological_nodes)
    # size = 10000
    # samples_Mynew = Mynew_forward_sampling(network, nodes).My_forward_sample(size, return_type='dataframe')
    # df_full = {'samples': samples_Mynew}
    # # save_excel(df_full, '../data/fulldata/child_10000.xlsx')
    # # save_excel(df_full, '../data/fulldata/alarm_500.xlsx')
    # # save_excel(df_full, '../data/fulldata/hepar2_10000.xlsx')
    # # save_excel(df_full, '../data/fulldata/andes_10000.xlsx')
    # # save_excel(df_full, '../data/fulldata/munin1_10000.xlsx')
    # save_excel(df_full, '../data/fulldata/link_10000.xlsx')
    #
    # # print('samples_Mynewo:\n', samples_Mynew, '\n')
    # samples_Mynew.loc[:, latent_vars] = np.nan
    # df_all = {'samples': samples_Mynew}
    #
    # # save_excel(df_all, '../data/missingdata/child_10000_disease_LungParench.xlsx')
    # # save_excel(df_all, '../data/missingdata/alarm_10000_TPR_CO.xlsx')
    # # save_excel(df_all, '../data/missingdata/hepar2_10000_surgery_fat.xlsx')
    # # save_excel(df_all, '../data/missingdata/andes_10000_SNode_3_SNode_4.xlsx')
    # # save_excel(df_all, '../data/missingdata/munin1_10000_MUDENS_MALOSS.xlsx')
    # save_excel(df_all, '../data/missingdata/link_10000_N73.xlsx')


    # sample_500 = read_excel('../data/network_500_location_age.xlsx')
    # print('sample_500:\n', sample_500)
    # #%%
    # nodes_store = ['amenities', 'neighborhood', 'location', 'children', 'size', 'schools', 'age', 'price']
    # size = 5000
    # samples_Mynew = Mynew_forward_sampling(network, nodes_store).My_forward_sample(size, return_type='dataframe')


    # # #%%
    # mis_var = []
    # sample_t = sampling_for_EM(network, mis_var)
    # df_all = {'samples': sample_t}
    # save_excel(df_all, '../data/fulldata/500/alarm_complete_500.xlsx')

    # #%%
    # sample_df = sampling_for_allsamples(network['V'], network['cardinality'])
    # df_all = {'samples': sample_df}
    # save_excel(df_all, '../data/fulldata/500/alarm_complete_500.xlsx')

    #%%
    bn_name = ['child', 'alarm', 'hepar2', 'andes', 'munin1', 'link']
    bn_data = ['../data/bn/child.bif', '../data/bn/alarm.bif', '../data/bn/hepar2.bif',
               '../data/bn/andes.bif', '../data/bn/munin1.bif', '../data/bn/link.bif']
    latent_nodes = [['disease', 'LungParench'], ['TPR', 'CO'], ['surgery', 'fat'], ['SNode_3', 'SNode_4'],
                    ['R_MYAS_APB_MUDENS', 'R_LNLW_APB_MALOSS'], ['N73_a_m', 'N73_a_f']]
    for i in range(6):
        reader = BIFReader(bn_data[i])
        network = reader.my_model()
        nodes = network['V']
        size = 500
        samples_Mynew = Mynew_forward_sampling(network, nodes).My_forward_sample(size, return_type='dataframe')
        df_full = {'samples': samples_Mynew}
        save_excel(df_full, '../data/fulldata/500' + bn_name[i] + '_' + str(size) + '.xlsx')

        # samples_Mynew.loc[:, latent_nodes[i]] = np.nan
        # df_all = {'samples': samples_Mynew}
        # save_excel(df_full, '../data/missingdata/'+bn_name[i]+'_'+str(size)+'.xlsx')