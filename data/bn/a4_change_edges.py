from pgmpy.sampling import BayesianModelSampling
from pgmpy.readwrite.BIF import BIFReader, BIFWriter
import pandas as pd
import numpy as np
from a1_plot_bn import plot
from graphviz import Digraph
from copy import deepcopy
import os


def save_excel(latent_vars, values, file_path):
    if not latent_vars:
        pass
    else:
        for i in range(len(latent_vars)):
            values[latent_vars[i]] = np.nan

    writer_excel = pd.ExcelWriter(file_path)
    # for key, value in df_all.items():
    #     value.to_excel(writer_excel, sheet_name=key)
    values.to_excel(writer_excel, sheet_name="1", index=True)
    writer_excel.save()




def generete_origin_net(data_path, net_name, num_sample):
    reader = BIFReader(data_path + "rename_" + net_name + ".bif", n_jobs=1)
    net = reader.get_model()
    values = BayesianModelSampling(net).forward_sample(int(num_sample), seed=42)
    # print(type(net.nodes))
    # print(type(values))
    # print(net.nodes())
    # print(net.edges())
    # exit(0)

    list_remove_edge_node = [str(i) for i in range(1, 21)]
    # print(list_remove_edge_node)

    remove_edges = []
    for edge in net.edges():
        if edge[0] in list_remove_edge_node or edge[1] in list_remove_edge_node:
            remove_edges.append(edge)
    print(remove_edges)

    # print(net.get_cpds('1'))
    # print(net.get_cpds('2'))

    add_edges = [('1', '17'), ('2', '17'), ('3', '17'), ('4', '17'),
                 ('5', '18'), ('6', '18'), ('7', '18'), ('8', '18'),
                 ('9', '19'), ('10', '19'), ('11', '19'), ('12', '19'),
                 ('13', '20'), ('14', '20'), ('15', '20'), ('16', '20')]

    # print(net.edges)
    # exit(0)
    net.remove_edges_from(remove_edges)
    # print(net.edges)
    net.add_edges_from(add_edges)


    with open(data_path + "rename_" + net_name + ".txt", "a") as fout:
        print(data_path + "rename_" + net_name + ".txt")
        lines = "edges" + '\n'
        edges = []
        for e in net.edges():
            edges.append([int(e[0]), int(e[1])])
        lines += str(edges) + '\n'
        print(lines)
        fout.write(lines)
    # exit(0)


    print(values)
    net.fit(values, state_names=net.states)
    # exit(0)

    # print(net.get_cpds('1'))
    # print(net.get_cpds('2'))
    latent_vars = []

    if os.path.exists(data_path + 'fulldata/' + str(num_sample)):
        print(data_path + 'fulldata/' + str(num_sample))
    else:
        os.mkdir(data_path + 'fulldata/' + str(num_sample))

    if net.check_model():
        BIF = BIFWriter(net)
        BIF.write_bif(data_path + 'fulldata/' + str(num_sample) + '/' + net_name + ".bif")

        G = Digraph('network')
        edges = net.edges()
        for a, b in edges:
            G.edge(a, b)
        G.render(data_path + 'fulldata/' + str(num_sample) + '/' + net_name + ".gv", view=False)
        save_excel(latent_vars, values, data_path + 'fulldata/' + str(num_sample) + '/' + net_name + '.xlsx')

        return values
    else:
        print("check is not done!!!!")



def generete_letent_net(data_path, latent_vars, add_edges, values, num_VQVAE):
    reader = BIFReader(data_path + 'fulldata/' + str(num_sample) + '/' + net_name + ".bif", n_jobs=1)
    net = reader.get_model()

    # print(net.edges)
    net.add_edges_from(add_edges)
    # print(net.edges)
    net.fit(values, state_names=net.states)

    if os.path.exists(data_path + 'missingdata/' + str(num_sample)):
        print(data_path + 'missingdata/' + str(num_sample))
    else:
        os.mkdir(data_path + 'missingdata/' + str(num_sample))

    if os.path.exists(data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars))):
        print(data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars)))
    else:
        os.mkdir(data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars)))

    if net.check_model():
        BIF = BIFWriter(net)
        BIF.write_bif(data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars)) + '/' + net_name + ".bif")
        G = Digraph('network')
        edges = net.edges()
        for a, b in edges:
            G.edge(a, b)
        G.render(data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars)) + '/' + net_name + ".gv", view=False)
        save_excel(latent_vars, values, data_path + 'missingdata/' + str(num_sample) + '/' + str(len(latent_vars)) + '/' + net_name + '.xlsx')
    else:
        print("check is not done!!!!")




bn_name = ['child', 'water', 'munin1', 'pigs']
# bn_name = ['child']

for net_name in bn_name:
    print(net_name)
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\" + net_name + "\\"
    # print(data_path)
    # exit(0)
    num_sample = 10000
    values = generete_origin_net(data_path, net_name, num_sample)

    print("L4")
    latent_vars = ['1', '2', '3', '4']
    generete_letent_net(data_path, latent_vars, [], deepcopy(values), num_VQVAE = 1)

    print("L8")
    latent_vars = ['1', '2', '3', '4', '5', '6', '7', '8']
    generete_letent_net(data_path, latent_vars, [], deepcopy(values), num_VQVAE = 2)

    print("L12")
    latent_vars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    generete_letent_net(data_path, latent_vars, [], deepcopy(values), num_VQVAE = 3)

    print("L16")
    latent_vars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    generete_letent_net(data_path, latent_vars, [], deepcopy(values), num_VQVAE = 4)
