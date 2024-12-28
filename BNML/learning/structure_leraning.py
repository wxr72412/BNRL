"""
    Tests NPN
    Borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
from torch.autograd import Variable
from learning.CPD_learning_process import CPD_learning
import numpy as np
from my_model.npn1 import GaussianNPN
# from torchviz import make_dot, make_dot_from_trace
import init_hyperparameters
import copy
import time

def generate_candidate_DAG_HC(BNML, num_k2_edges = 100):
    list_candidate_DAG = []
    list_candidate_changed_node = [] # 表示父节点发生变化的节点
    # print("edges:")
    # print(BNML.edges)
    # print("constraint_init_edges:")
    # print(BNML.constraint_init_edges)
    # print("constraint_may_edges:")
    # print(BNML.constraint_may_edges)
    # exit(0)
    # 加边
    candidate_DAG = None
    for i in range(0, BNML.num_nodes):
        for j in range(0, BNML.num_nodes):
            # print(i, j)
            if BNML.constraint_may_edges[i][j] == 0 or i == j or BNML.edges[i][j] == 1 or sum(BNML.edges[:, j]) >= num_k2_edges: # 节点自身，或者i和j之间已有边则跳过
                # print("节点自身，或者i和j之间已有边则跳过")
                continue
            else:
                candidate_DAG = BNML.edges.copy() # A是B的深复制
                candidate_DAG[i][j] = 1
                # print(candidate_DAG)
                if BNML.findcircle(candidate_DAG) == 0: # 0表示无环，1表示有环
                    list_candidate_DAG.append(candidate_DAG)
                    list_candidate_changed_node.append(j)
                    # print(list_candidate_DAG)
    return list_candidate_DAG, list_candidate_changed_node


def structure_train_process(BNML, para, input, input_other):

    epoch = 0
    while 1:
        epoch += 1
        ts = time.time()
        # 生成候选模型
        list_candidate_DAG, list_candidate_changed_node = generate_candidate_DAG_HC(BNML, para['num_k2_edges'])
        # print("list_candidate_DAG:")
        # print(list_candidate_DAG)
        # print("list_candidate_changed_node:")
        # print(list_candidate_changed_node)
        if list_candidate_DAG == []:
            print('No candidate!')

        current_optimal_model = None
        current_optimal_BIC_increased = float(0)

        for index in range(len(list_candidate_DAG)):
            # 第i个节点
            i = list_candidate_changed_node[index]
            candidate_DAG = list_candidate_DAG[index]
            print()
            print('-------Structure Learning Epoch: {}, 第 {} 个候选模型, 第 {} 个节点-------'.format(epoch, index+1, i+1))
            print(candidate_DAG)
            # print(BNML)
            # candidate_model = BNML
            candidate_model = BNML.copy_BNML_without_CPD()
            # candidate_model = copy.deepcopy(BNML)
            # print(BNML)
            # print(candidate_model)
            # exit(0)

            candidate_model.edges = candidate_DAG
            # print(BNML.edges)
            # print(candidate_model.edges)
            candidate_model.list_NPN[i] = copy.deepcopy(BNML.list_NPN[i])
            # print(BNML.list_NPN)
            # print(candidate_model.list_NPN)
            # exit(0)

            BIC_loglikelihood_node_i = CPD_learning(candidate_model, i, para, input, input_other)
            BIC_penalty_node_i = candidate_model.independent_parameters_node_i(i) / 2 * np.log(para['train_size'])
            BIC_node_i = BIC_loglikelihood_node_i - BIC_penalty_node_i

            candidate_model.list_BIC_loglikelihood[i] = BIC_loglikelihood_node_i
            candidate_model.list_BIC_penalty[i] = BIC_penalty_node_i
            candidate_model.list_BIC[i] = BIC_node_i
            # print(BIC_loglikelihood_node_i)
            # print(BIC_penalty_node_i)
            # print(BIC_node_i)
            # print(BNML.list_BIC[i])

            print(candidate_model.list_BIC)
            print(BNML.list_BIC)
            BIC_increased = candidate_model.list_BIC[i] - BNML.list_BIC[i]
            # print('BIC_increased:')
            # print(BIC_increased)
            # print('current_optimal_BIC_increased:')
            # print(current_optimal_BIC_increased)
            if BIC_increased > current_optimal_BIC_increased: # 将当前模型作为最优模型
                current_optimal_BIC_increased = BIC_increased
                # print('current_optimal_BIC_increased:')
                # print(current_optimal_BIC_increased)
                current_optimal_model = copy.deepcopy(candidate_model)
                # print(current_optimal_model)
            del candidate_model
            # print(current_optimal_model)
            # print(current_optimal_model.list_BIC_loglikelihood[i])
            # print(current_optimal_model.list_BIC_penalty[i])
            # print(current_optimal_model.list_BIC[i])
            # print(current_optimal_model.list_NPN)

        # exit(0)
        print("time=", "{:.4f}".format(time.time() - ts))

        if current_optimal_BIC_increased == 0: # 收敛，返回结果
            return BNML
        else:
            BNML.edges = current_optimal_model.edges
            print('Current optimal model:')
            print(BNML.edges)
            # print(current_optimal_model.list_NPN)
            for i in range(0, BNML.num_nodes):
                if current_optimal_model.list_NPN[i] != None:
                    BNML.list_NPN[i] = copy.deepcopy(current_optimal_model.list_NPN[i])
                    BNML.list_BIC_loglikelihood[i] = current_optimal_model.list_BIC_loglikelihood[i]
                    BNML.list_BIC_penalty[i] = current_optimal_model.list_BIC_penalty[i]
                    BNML.list_BIC[i] = current_optimal_model.list_BIC[i]
                    BNML.output_CPD_node_i(i, para)
        del current_optimal_model
        # print(BNML.edges)
        # print(BNML.list_NPN)
        # exit(0)

        if epoch == para['structure_learning_max_iter_num']:
            print('Structure learning epoch: {}, and epoch has been equal to convergence_iter_num: {}'.format(epoch, para['structure_learning_max_iter_num']))
            return BNML

