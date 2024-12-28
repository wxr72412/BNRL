import networkx as nx
from pgmpy.readwrite import BIFReader, BIFWriter
from graphviz import Digraph
from pgmpy.inference import VariableElimination, ApproxInference
import time
import os

# reader = BIFReader('data/network.bif')  #network
# reader = BIFReader('data/dataset/alarm.bif')  #Alarm
# reader = BIFReader('../bn/child.bif')  # Large BN
# reader = BIFReader('data/dataset/andes.bif')  #Very large BN
# !rm friends.bif
# network = reader.get_model()


from pgmpy.models import BayesianNetwork

import numpy as np
import pandas as pd

#################################### 注意 ############################################
read_cpd_rould_decimal = None
read_cpd_rould = None
#################################### 注意 ############################################

####################################################################################################################
####################################################################################################################

def rename(net_name, data_path):
    # from BN.my_BIF import BIFReader, BIFWriter
    # reader1 = BIFReader(net_name + '.bif')
    # model1 = reader1.my_model()
    #
    # reader2 = BIFReader("rename_" + net_name + ".bif")
    # model2 = reader2.my_model()
    # print(model1['V'])
    # for var1, var2 in zip(model1['V'], model2['V']):
    #     print(model1['cpds'][var1]) # <class 'list'>
    #     print(model2['cpds'][var2])  # <class 'list'>
    #     print()
    #
    #     print(model1['cpds_df'][var1])
    #     print(model2['cpds_df'][var2])
    #     print()
    #
    #     print(model1['learn_cpds_df'][var1])
    #     print(model2['learn_cpds_df'][var2])
    #     print()
    #
    # exit(0)

    from pgmpy.readwrite import BIFReader, BIFWriter
    reader1 = BIFReader(data_path + net_name + '.bif', n_jobs=1)
    model1 = reader1.get_model()

    reader2 = BIFReader(data_path + "rename_" + net_name + ".bif", n_jobs=1)
    model2 = reader2.get_model()

    reader4 = BIFReader(data_path + "rename2_" + net_name + ".bif", n_jobs=1)
    model4 = reader4.get_model()

    inference1 = VariableElimination(model1)
    t1 = time.time()
    # phi_query = inference1.query(['R_LNLW_APB_DENERV'])
    phi_query = inference1.query(variables = ['R_LNLW_APB_DENERV'], evidence = {'R_LNLW_MED_SEV': 'MILD', 'R_LNLW_MED_TIME': 'ACUTE', 'R_LNLW_MED_PATHO': 'DEMY'})
    t2 = time.time()
    print(phi_query)
    print(t2-t1)

    inference2 = VariableElimination(model2)
    t1 = time.time()
    # phi_query = inference2.query(['3'])
    phi_query = inference2.query(variables=['72'], evidence={'71': '1', '70': '0', '66': '0'})
    t2 = time.time()
    print(phi_query)
    print(t2 - t1)

    inference4 = VariableElimination(model4)
    t1 = time.time()
    # phi_query = inference4.query(['3'])
    phi_query = inference4.query(variables=['72'], evidence={'71': '1', '70': '0', '66': '0'})
    t2 = time.time()
    print(phi_query)
    print(t2 - t1)




# bn_name = ['child', 'water', 'munin1', 'pigs']
bn_name = ['munin1']

for net_name in bn_name:
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\" + net_name + "\\"
    print(data_path)
    rename(net_name, data_path)
