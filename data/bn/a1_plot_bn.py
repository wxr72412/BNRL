import networkx as nx
from pgmpy.readwrite import BIFReader
from graphviz import Digraph
import os


# print(os.environ['PATH'])
# exit(0)

def plot(net_name):
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\" + net_name + "\\"
    print(data_path)
    # exit(0)

    # reader = BIFReader('data/network.bif')  #network
    # reader = BIFReader('data/dataset/alarm.bif')  #Alarm
    # reader = BIFReader('../bn/child.bif')  # Large BN
    # reader = BIFReader('data/dataset/andes.bif')  #Very large BN
    # !rm friends.bif
    # reader = BIFReader(net_name + ".bif")
    reader = BIFReader(data_path + net_name + ".bif", n_jobs=1)
    network = reader.get_model()

    G = Digraph('network')
    # nodes = network.nodes()
    # print(nodes)
    # exit(0)

    # nodes_sort = list(nx.topological_sort(network))
    # print('nodes_num:', len(nodes_sort))
    # print('nodes_sort:', nodes_sort)
    # exit(0)

    edges = network.edges()
    # print('\nedges_num:', len(edges))
    # print('\nedges:', edges)

    for a, b in edges:
        G.edge(a, b)
    # var_card = {node: network.get_cardinality(node) for node in nodes_sort}
    # print('var_card:', var_card)
    # var_card=dict(zip(cpd.variables, cpd.cardinality))

    G.render(data_path + net_name + ".gv", view=False)
    # G
    # print(len(nodes))



# bn_name = ['child', 'water', 'munin1', 'pigs']
# for net_name in bn_name:
#     plot(net_name)