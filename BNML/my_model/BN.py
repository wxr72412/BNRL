import torch
import pandas as pd
from os import path
from torch.utils.data import Dataset
import numpy as np
from my_model.npn1 import GaussianNPN, vanillaNN
from torch.autograd import Variable
import init_hyperparameters
from prediction.MAP import NPN_MAP_process
import init_hyperparameters
import os
from my_model.BN_init_theta import init_theta



def load_BN(para):
    data_path = os.getcwd()
    data_path += '\\data\\'
    data_path += para['data_file'] + '\\' + para['BN_file']
    print(data_path)
    return torch.load(data_path)

def load_BN_file_name(para, file_name):
    data_path = os.getcwd()
    data_path += '\\data\\'
    data_path += para['data_file'] + '\\' + file_name
    print(data_path)
    return torch.load(data_path)

def load_BN_files(para, type, other = ''):
    data_path = None
    if type == "C":
        data_path = other \
                    + para['BN_file'] + '-C'
    elif type == "D":
        data_path = other \
                    + para['BN_file'] + '-D'
    elif type == "pre":
        data_path = other \
                    + para['BN_file'] + '-pre'
    print(data_path)
    return torch.load(data_path)

class BN():
    def __init__(self, num_nodes, list_node_type, list_node_label, list_cardinalities, list_latent_variables,
                 list_correspond_observed_variables, init_edges, constraint_may_edges, para):
        print("---------------创建BN---------------------")
        # print("---------------创建BN---------------------")
        # print("---------------创建BN---------------------")
        assert (num_nodes > 0)
        assert (len(list_node_type) == num_nodes)
        assert (len(list_node_label) == num_nodes)
        assert (len(list_cardinalities) == num_nodes) # 列表: 0表示连续变量，1~n表示离散变量的势
        assert (len(list_latent_variables) == num_nodes)
        assert (len(list_correspond_observed_variables) == num_nodes)
        assert (init_edges.shape == (num_nodes, num_nodes))
        assert (constraint_may_edges.shape == (num_nodes, num_nodes))
        assert (self.findcircle(init_edges) == 0) # 0表示无环，1表示有环
        self.para = para
        self.num_nodes = num_nodes # BNML变量个数
        self.list_node_type = list_node_type # 变量类型
        self.list_node_label = list_node_label # 变量标签
        self.list_cardinalities = list_cardinalities # 列表: 变量的势
        self.list_latent_variables = list_latent_variables # 列表: 隐变量
        self.list_correspond_observed_variables = list_correspond_observed_variables # 列表: 隐变量
        self.edges = init_edges # DAG结构
        self.constraint_init_edges = init_edges # 初始DAG结构作为必须存在边的约束
        self.constraint_may_edges = constraint_may_edges # 可能存在边的约束

        if para['data_file'] == 'ml-1m' or para['data_file'] == 'bone-marrow':
            self.list_init_theta = init_theta(self, self.para)


        self.list_BIC_loglikelihood = [float("-inf") for i in range(0, self.num_nodes)] # 存放个变量的对数似然
        self.list_BIC_penalty = [float(0) for i in range(0, self.num_nodes)] # 存放个变量的罚项
        self.list_BIC = [float("-inf") for i in range(0, self.num_nodes)] # 存放每个变量的family_BIC

        self.list_evidence_node = [] # 列表: 证据变量
        self.list_search_node = [] # 列表: 查询变量

        self.list_parent_variables = []


        # print(self.list_evidence_node)
        # print(self.list_search_node)
        # exit(0)

        self.latent = 0 # 隐变量标记，如果BNML存在隐变量则为1，反之为0
        if 1 in list_latent_variables: # There is at least one latent variable in BNML.
            self.latent = 1

        self.list_dependent_latent = np.zeros((num_nodes), dtype = int) # 列表: 变量是否依赖于隐变量 whether a variable and its parents are independent of the latent variables or not.
        # print(self.list_dependent_latent)

        self.sum_cardinalities = sum(list_cardinalities)
        # print(self.sum_cardinalities)

        self.list_NPN = [None for i in range(0, self.num_nodes)] # 每个变量对应一个NPN

        self.list_parent_variables = [self.list_parents(i) for i in range(self.num_nodes)]


    def create_CPD(self, device = "cpu", i = None):
        if i == None:
            for i in range(self.num_nodes): # 0 ~ num_nodes-1
                para = init_hyperparameters.para.copy()
                para['output_type'] = self.list_node_type[i]
                print(self.para['layers_NPN'])
                self.list_NPN[i] = GaussianNPN(self.sum_cardinalities, self.list_cardinalities[i], self.list_node_type[i], self.para['layers_NPN'], self.para).to(device)
                # self.list_NPN[i] = vanillaNN(self.sum_cardinalities, self.list_cardinalities[i], self.list_node_type[i], [128, 32, 8], self.para).to(device)
                print(self.list_NPN[i])
                print(next(self.list_NPN[i].parameters()).is_cuda)
        else:
            # print(type(i))
            assert(type(i) == int)
            para = init_hyperparameters.para.copy()
            para['output_type'] = self.list_node_type[i]
            self.list_NPN[i] = GaussianNPN(self.sum_cardinalities, self.list_cardinalities[i], self.list_node_type[i], [100, 100], para).to(device)
            # self.list_NPN[i] = vanillaNN(self.sum_cardinalities, self.list_cardinalities[i], self.list_node_type[i], [100, 100], para).to(device)
        # print(self.list_NPN)

    def Num_parents_value_combinations(self, i):
        Num_parents_value_combinations = 1
        for q in self.list_cardinalities * self.edges[:, i]:
            if q > 0:
                Num_parents_value_combinations *= q
        return Num_parents_value_combinations

    def independent_parameters_node_i(self, i):
        Num_parents_value_combinations = self.Num_parents_value_combinations(i)
        d = Num_parents_value_combinations * (self.list_cardinalities[i]-1)
        return d

    def parameters_node_i(self, i):
        Num_parents_value_combinations = self.Num_parents_value_combinations(i)
        d = Num_parents_value_combinations * self.list_cardinalities[i]
        return d

    def output_CPD_node_i(self, i, para):
        # print(self.list_cardinalities)
        # print(self.edges)
        # print(self.edges[:, i])
        # list_parent_cardinalities = self.list_cardinalities * self.edges[:, i]
        # print(list_parent_cardinalities)
        Num_parents_value_combinations = self.Num_parents_value_combinations(i)
        # print(Num_parents_value_combinations)
        x = torch.FloatTensor(Num_parents_value_combinations, self.sum_cardinalities).zero_().to(para['device'])
        for index in range(Num_parents_value_combinations):
            temp = index
            for j in range(self.num_nodes):
                reverse_j = self.num_nodes - 1 - j
                # print(reverse_j)
                if self.edges[reverse_j][i] == 1: # 给父节点对应的行赋值
                    q = torch.tensor([temp % self.list_cardinalities[reverse_j] + 1])
                    # print(temp)
                    # print(self.list_cardinalities[reverse_j])
                    # print(q)
                    temp = (temp / (self.list_cardinalities[reverse_j])).__int__()
                    # print(temp)
                    # print(torch.FloatTensor(1, self.list_cardinalities[reverse_j]).zero_())
                    xx = torch.FloatTensor(1, self.list_cardinalities[reverse_j]).zero_().scatter_(1, q.to(int).unsqueeze(1)-1, 1)
                    # print(xx)
                    x[index][sum(self.list_cardinalities[:reverse_j]):sum(self.list_cardinalities[:reverse_j+1])] = xx
                    # print(x)
        # print(x[99])
        # exit(0)
        # print()
        data_s = torch.zeros(x.size()).to(para['device'])
        # print(data_s)
        # exit(0)
        output = self.list_NPN[i]((x, data_s))
        # print('---------CPD of Node: {}-----------'.format(i+1))
        # print(output[0])
        return output[0], output[1]



    # def copy_CPD(self, i = None, BNML = None):
    #     assert(BNML != None)
    #     if i == None:
    #         for i in range(self.num_nodes): # 0 ~ num_nodes-1
    #             self.list_NPN[i] = BNML.list_NPN[i].copy()
    #     else:
    #         # print(type(i))..
    #         assert(type(i) == int)
    #         print(BNML.list_NPN[i])
    #         print(BNML.list_NPN[i].copy())
    #         exit(0)
    #         self.list_NPN[i] = BNML.list_NPN[i].copy()



    def copy_BNML_without_CPD(self):
        return BN(self.num_nodes, self.list_node_type, self.list_node_label, self.list_cardinalities, self.list_latent_variables, self.edges, self.constraint_may_edges)

    def dfs(self, G, i, color):
        r = len(G)
        color[i] = -1
        have_circle = 0
        for j in range(r):	# 遍历当前节点i的所有邻居节点
            if G[i][j] != 0:
                if color[j] == -1:
                    have_circle = 1
                elif color[j] == 0:
                    have_circle = self.dfs(G, j, color)
        color[i] = 1
        return have_circle

    def findcircle(self, G):
        # color = 0 该节点暂未访问
        # color = -1 该节点访问了一次
        # color = 1 该节点的所有孩子节点都已访问,就不会再对它做DFS了
        r = len(G)
        color = [0] * r
        have_circle = 1
        for i in range(r):	# 遍历所有的节点
            if color[i] == 0:
                have_circle = self.dfs(G, i, color)
                if have_circle == 0:
                    break
        return have_circle

    def list_parents(self, i):
        list_parents = []
        for j in range(self.num_nodes):  # 0 ~ num_nodes-1
            if self.edges[j][i] == 1:  # 父节点
                list_parents.append(j)
        return list_parents

    def list_dependent_latent_node(self, i):
        list_dependent_latent_node = []
        for j in range(self.num_nodes):  # 0 ~ num_nodes-1
            if self.edges[j][i] == 1 or i == j:  # 父节点或子节点j
                if self.list_latent_variables[j] == 1:
                    list_dependent_latent_node.append(j)
        return list_dependent_latent_node

    def list_parent_nodes(self, i):
        list_parent_nodes = []
        for j in range(self.num_nodes):  # 0 ~ num_nodes-1
            if self.edges[j][i] == 1:  # 父节点
                list_parent_nodes.append(j)
        return list_parent_nodes

    def list_child_nodes(self, i):
        list_child_nodes = []
        for j in range(self.num_nodes):  # 0 ~ num_nodes-1
            if self.edges[i][j] == 1:  # 父节点
                list_child_nodes.append(j)
        return list_child_nodes

    def save_model(self, para):
        data_path = os.getcwd()
        data_path += '\\data\\'
        data_path += para['data_file'] + '\\' + para['BN_file']
        torch.save(self, data_path)
        print("------------Saving model is done!-------------------")

    def save_BN_file_name(self, para, file_name):
        data_path = os.getcwd()
        data_path += '\\data\\'
        data_path += para['data_file'] + '\\' + file_name
        torch.save(self, data_path)
        print("------------Saving model is done!-------------------")

    def save_BN_files(self, para, type, other = ''):
        data_path = None
        if type == "C":
            data_path = other \
                        + para['BN_file'] + '-C'
        elif type == "D":
            data_path = other \
                        + para['BN_file'] + '-D'
        elif type == "pre":
            data_path = other \
                        + para['BN_file'] + '-pre'
        print(data_path)
        torch.save(self, data_path)
        print("------------Saving model is done!-------------------")





if __name__ == '__main__':

    num_nodes = 2 # BNML变量个数
    list_node_type = ["D", "D"] # 列表: C表示连续变量，D表示离散变量
    list_cardinalities = [2, 2] # 列表: 1表示连续变量的势，1~n表示离散变量的势
    list_latent_variables = [0, 0] # 列表: 0表示显变量，1表示隐变量
    list_edges = [[1, 2], [2, 1]] # 列表: 初始边的集合

    init_edges = np.zeros((num_nodes, num_nodes), dtype = int, order='C') # 初始边的邻接矩阵
    for e in list_edges:
        init_edges[e[0]-1][e[1]-1] = 1
    constraint_may_edges = init_edges
    # print(constraint_may_edges)
    # print(constraint_may_edges[0,:])
    # print(constraint_may_edges[:,1]) #
    # exit(0)




    BNML = BN(num_nodes, list_node_type, list_cardinalities, list_latent_variables, init_edges, constraint_may_edges)
    print(BNML.edges)
    print(BNML.findcircle(BNML.edges))
    print(BNML.edges[:, 1])
    #
    # para = init_hyperparameters.para
    # train_loader = BNML.load(para)
    # print(train_loader)
    # # for t in train_loader:
    # #     print(t)
    exit(0)




