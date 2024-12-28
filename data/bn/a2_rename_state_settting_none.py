from graphviz import Digraph
import os
import init_para
# net_bif = 'child.bif'
# net_bif = 'alarm.bif'
# net_bif = 'hepar2.bif'
# net_bif = 'andes.bif'
# net_bif = 'munin1.bif'
# net_bif = 'link.bif'

#################################### 注意 需要手动再设置一次， 否则出现概率和！=1的错误 ############################################
# init_para.read_cpd_rould_decimal = None
# init_para.read_cpd_rould = None
#################################### 注意， 否则出现概率和！=1的错误 ############################################


def rename(net_name, data_path):
    from BN.my_BIF import BIFReader, BIFWriter
    print(data_path + net_name + ".bif")
    reader = BIFReader(data_path + net_name + ".bif")
    net = reader.my_model()

    writer = BIFWriter(net)

    # print(net['learn_cpds_df']["R_LNLW_MED_SEV"])
    # exit(0)

    from copy import deepcopy
    dict_V = {}
    dict_states = deepcopy(net['states'])
    i = 1
    net_V = net['V']

    # print(net_V)
    if net_name == 'water':
        net_V.remove("CKND_12_45")
        net_V.remove("CKNN_12_30")
        net_V.remove("CKND_12_30")
        net_V.remove("CKNI_12_30")
        net_V.insert(0, "CKND_12_45") # 19
        net_V.insert(0, "CKNN_12_30") # 18
        net_V.insert(0, "CKND_12_30") # 17
        net_V.insert(0, "CKNI_12_30") # 14
    elif net_name == 'munin1':
        net_V.remove("R_LNLW_APB_DENERV")
        net_V.remove("R_LNLW_MED_SEV")
        net_V.remove("R_LNLW_MED_TIME")
        net_V.remove("R_LNLW_MED_PATHO")
        net_V.insert(0, "R_LNLW_APB_DENERV") # 72
        net_V.insert(0, "R_LNLW_MED_SEV") # 71
        net_V.insert(0, "R_LNLW_MED_TIME") # 70
        net_V.insert(0, "R_LNLW_MED_PATHO") # 66


    # print(net_V)
    # exit(0)



    for var in net_V:
        dict_V[var] = str(i)
        del dict_states[var]
        dict_states[str(i)] = [str(c) for c in range(net['cardinality'][var])]
        # exit(0)
        i += 1



    print(net_name)
    print('V')
    print(net['V'])
    print('VN')
    print(net['VN'])
    print(dict_V)
    print('states')
    print(net['states'])

    print('cardinality')
    print(net['cardinality'])
    print()

    # dict_states = print(net['learn_cpds_df']["HypoxiaInO2"])
    # exit(0)

    writer.write_bif_rename(data_path + "rename_" + net_name + ".bif", dict_V, dict_states)
    # exit(0)

    with open(data_path + "rename_" + net_name + ".txt", "w") as fout:
        lines = ""
        lines += net_name + '\n' + '\n'

        for k1, k2 in zip(dict_V, net['states']):
            # print(k1)
            # print(k2)
            # print(dict_V[k1])
            lines += dict_V[k1] + ': ' + k1 + ' (' + str(net['cardinality'][k1]) + ')' + '   '
            # print(lines)

            # print(net['states'])
            # print(dict_states)

            lines += str(dict_states[dict_V[k1]]) + '   ' + str(net['states'][k1])
            # print(lines)
            # exit(0)
            lines +='\n'
        lines += '\n'
        lines += 'cardinality' + '\n'
        for k1 in dict_V:
            lines += str(net['cardinality'][k1]) + ', '
        lines += '\n'
        # print(lines)
        fout.write(lines)

    # exit(0)


    # reader = BIFReader(net_name + ".bif")
    # reader = BIFReader("rename_" + net_name + ".bif")
    # net = reader.my_model()
    # writer = BIFWriter(net)
    # writer.write_bif("rename1_" + net_name + ".bif")

    def plot(net_name):
        from pgmpy.readwrite import BIFReader as Reader
        reader = Reader(data_path + net_name + ".bif", n_jobs = 1)
        network = reader.get_model()
        G = Digraph('network')
        edges = network.edges()
        for a, b in edges:
            G.edge(a, b)
        G.render(data_path + net_name + ".gv", view=False)

    plot("rename_" + net_name)

    # from pgmpy.readwrite import BIFReader, BIFWriter
    # reader = BIFReader(data_path + "rename_" + net_name + ".bif", n_jobs=1)
    # net = reader.get_model()
    # writer = BIFWriter(net)
    # writer.write_bif("rename2_" + net_name + ".bif")


# bn_name = ['child', 'water', 'munin1', 'pigs']
bn_name = ['water', 'munin1']
# bn_name = ['munin1']

# bn_name = ['child']



for net_name in bn_name:
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\" + net_name + "\\"
    print(data_path)
    rename(net_name, data_path)

    from pgmpy.readwrite import BIFReader, BIFWriter
    reader = BIFReader(data_path + "rename_" + net_name + ".bif", n_jobs = 1)
    net = reader.get_model()
    writer = BIFWriter(net)
    writer.write_bif(data_path + "rename2_" + net_name + ".bif")





