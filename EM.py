from __future__ import division

import pgmpy.readwrite

import para_init
from BN.my_BIF import BIFReader, BIFWriter
from BN import my_bayesnet
import numpy as np
import pandas as pd
import time
import para_init
from metrics import MSE, KL

# 显示所有列
from BN.my_bayesnet import save_excel

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 显示行的长度
pd.set_option('display.width', 1000)


# Setup the network
def setup_network(net_bif, dat_records):
    # Parsing the network from .bif format
    print("0: Reading Network . . . ")
    print(net_bif)
    reader = BIFReader(net_bif)
    net = reader.my_model()
    # for v in net['V']:
    #     # print(net['cpds'][v])
    #     # print(net['cpds_df'][v])
    #     print(net['learn_cpds_df'][v])
    # print(net['learn_cpds_df'])

    # reader1 = pgmpy.readwrite.BIFReader(net_bif)
    # net1 = reader1.get_model()
    # for cpd in net1.get_cpds():
    #     print(cpd)
    # Get data from record.dat
    # print("1: Getting data from records . . . ")
    df = my_bayesnet.read_excel_data(dat_records)
    # print(df)
    # exit(0)

    # net.setdefault('learn_cpds_df_wts', deepcopy(net['learn_cpds_df']))
    # Get the index of nodes which have missing value in each row
    print("3: Getting missing data indexes . . . ")
    mis_index, mis_var = get_missing_index(df)
    print(mis_index)
    print(mis_var)
    # exit(0)

    # Initialise parameters
    print("2: Initialising parameters . . . ")
    net = my_bayesnet.my_init_param(df, net, mis_var)
    # print(net['V'])
    # for v in net['V']:
        # print(net['cpds'][v])
        # print(net['cpds_df'][v])
        # print(net['learn_cpds_df'][v])
    # exit(0)



    return net, df, mis_index


# List of the indices of nodes which have missing values in each data point;
# equal to -1 if no value is missing
def get_missing_index(df):  # get the index of column of df (dataset)
    """
    param df: df is read from record.dat
                       "Hypovolemia"  "StrokeVolume"  ...  "VentLung"  "Intubation"
            0                1.0             1.0  ...         NaN           NaN
            1                1.0             1.0  ...         NaN           NaN
            2                1.0             1.0  ...         NaN           NaN
            3                0.0             0.0  ...         NaN           NaN
            4                0.0             0.0  ...         NaN           NaN
        return e.g., [2,3] represents the index of the latent variable
    """

    row_mis = df.iloc[0, :].T
    # return the index of nonzero, e.g., array([[4], [5], [6]])
    # print('np.isnan:\n', np.isnan(np.asarray(row_mis)))
    mis_index = np.argwhere(np.isnan(np.asarray(row_mis)))
    mis_index = [int(i) for i in mis_index]
    mis_var = [df.columns.tolist()[i] for i in mis_index]
    return mis_index, mis_var


# Expectation Step
def Expectation(bn, df, mis_index):
    """
       Input:
           bn - Bayesian Network
           df - Data table
           mis_index - array of missing indices corresponding to the Data table 'df'
       Output:
           new_df - each missing value in a row replaced by the possible values variable can take

               original row: VentTube is NaN
               Failure"  "LVEDVolume"  "PCWP"  "CVP"  "History"  "MinVolSet"  "VentMach"  "Disconnect"  "VentTube"
               1.0           1.0     1.0    1.0        1.0          1.0         2.0           1.0         NaN
               new_df: (pd.concat)
               Failure"  "LVEDVolume"  "PCWP"  "CVP"  "History"  "MinVolSet"  "VentMach"  "Disconnect"  "VentTube" “wts”
               1.0           1.0     1.0    1.0        1.0          1.0         2.0           1.0         0.0       0.3
               1.0           1.0     1.0    1.0        1.0          1.0         2.0           1.0         1.0       0.2
               1.0           1.0     1.0    1.0        1.0          1.0         2.0           1.0         2.0       0.3
               1.0           1.0     1.0    1.0        1.0          1.0         2.0           1.0         3.0       0.3

           new_weights - array of weights assigned to each data point
               new_weights: [5.794085565677744e-07, 2.409855880627196e-06, 0.9999954002622516, 1.6104733111558328e-06]
       """
    # print(df)
    # print(mis_index)
    # exit(0)
    new_new_df = pd.DataFrame()
    # print(new_new_df)
    # new_df_list = []
    for i in range(df.shape[0]):  # i represents the i th row
        # t1 = time.time()
        # print(20*'**')
        """
        row = pd.DataFrame(df.loc[i, :]).T:

                      size  neighborhood  children  schools  amenities  location  age  price
                0       1             0         0        1        NaN        NaN  NaN    2.0
                1       2             1         0        0        NaN        NaN  NaN    0.0
                2       2             1         1        1        NaN        NaN  NaN    1.0
                3       0             1         1        0        NaN        NaN  NaN    1.0
                4       1             0         0        1        NaN        NaN  NaN    1.0
        """
        new_df = pd.DataFrame()
        # print(new_df)
        row = pd.DataFrame(df.loc[i, :]).T
        # print('row:\n', row)
        # exit(0)
        # mis_index = [1, 5],1 and 5 represents having two the latent variables, their index being 1 and 5 respectively
        for j in mis_index:
            X = df.columns.tolist()[j]
            # print(X)
            # X = bn['V'][mis_index[j]]  # the name of variable having missing value
            if len(new_df.index) == 0:
                for n in range(bn['cardinality'][X]):
                    row.iloc[0, j] = n  # the value of X in row is replaced by current n
                    new_df = pd.concat([new_df, row])  # concat the row (df type) to new_df
                new_df = new_df.reset_index(drop=True)
                # print(new_df)
                # exit(0)
            else:
                for r in range(new_df.shape[0]):
                    new_row = pd.DataFrame(new_df.loc[0, :]).T  # acquire the row of new_df
                    # print('new_row.loc[0, {j}]:{v}'.format(j=mis_index[j], v=new_row.iloc[0, mis_index[j]]))
                    if np.isnan(new_row.iloc[0, j]):  # judge the new_row whether the mis_index[j] has NaN
                        # if the new_row has NaN element, traverse the cardinality of X

                        for n in range(bn['cardinality'][X]):
                            new_row.iloc[0, j] = n
                            new_df = pd.concat([new_df, new_row])
                            # reindex the row of new_df in order to drop the new_df.loc[0, :] conveniently
                            new_df = new_df.reset_index(drop=True)
                        new_df.drop([0], axis=0, inplace=True)  # delete the 1st row in the new_df
                        # reindex the row of new_df in order to find the new row (new_df.loc[0, :]) conveniently
                        new_df = new_df.reset_index(drop=True)
                        # print('new_df:/n', new_df)
                    else:
                        break
        new_new_df = pd.concat([new_new_df, new_df])
        new_new_df = new_new_df.reset_index(drop=True)
        # print(new_new_df)
        # t2 = time.time()
        # print("time1-1=", "{:.4f}".format(t2 - t1))
    # print(new_new_df)
    # exit(0)
    new_weights = []
    # dict(orient="records") like [{'"LVEDVolume"': 0, '"Hypovolemia"': 1...}, {'"LVEDVolume"': 22, ...}]
    mydict = new_new_df.to_dict(orient='records')
    # print(mydict)
    # exit(0)
    # print('new_new_df:\n', new_new_df)
    # new_df_list = []

    product_car_latent = int(new_new_df.shape[0] / df.shape[0])
    # print(product_car_latent)
    # exit(0)
    for n in range(df.shape[0]):  # i represents the i th row
        sum = 0.0
        weights = []
        for c in range(product_car_latent):  # i represents the i th row
            i = n * product_car_latent + c
            # print(i)
            # t1 = time.time()
            # print(20 * '**')
            # print('mydict:', mydict[i])
            dist_X_weight = []
            for j in range(len(bn['V'])):
                # print(20 * '---')
                X = bn['V'][j]
                # print('current variable:', X)
                X_pares = bn['parents'][X] + [X]  # X_pares denotes 'pares + X'
                # print(X_pares)
                X_cpds = bn['learn_cpds_df'][X]
                # print(X_cpds)
                t1_1 = time.time()
                if not X_pares:  # if current variable has no parents
                    X_cpds = X_cpds.loc[X_cpds[X] == mydict[i][X], 'p']
                    # print('x_cpds:\n', X_cpds.values[0])
                    dist_X_weight.append(X_cpds.values[0])
                    # print('dist_X_weight:', dist_X_weight)
                else:
                    # temp_X_cpds = pd.DataFrame()
                    for cur_X in X_pares:
                        # print(50*'***')
                        # print('cur_X:', cur_X)
                        # print('x_cpds:\n', X_cpds)
                        # print('my_dict[i]:\n', mydict[i])
                        X_cpds = X_cpds.loc[X_cpds[cur_X] == mydict[i][cur_X]]
                        # print('retrive_X_cpds:\n', X_cpds)
                    dist_X_weight.append(X_cpds.loc[:, 'p'].values[0])
                    # print('dist_X_weight:', dist_X_weight)
                # t1_2 = time.time()
                # print("time1-2-1=", "{:.4f}".format(t1_2 - t1_1))
                # exit(0)
            # print('dist_X_weight:', dist_X_weight)
            dist_X_weight_product = np.prod(dist_X_weight)
            # print('dist_X_weight_product:', dist_X_weight_product)
            weights.append(dist_X_weight_product)
        weights = np.array(weights)
        # print(weights)
        sum = + np.sum(weights)
        # print(sum)
        if sum != 0:
            weights = (weights / sum)
        weights.tolist()
        # print(weights)
        # exit(0)
        new_weights.extend(weights)

    # for i in range(new_new_df.shape[0]):  # i represents the row of df
    #     # print(i)
    #     t1 = time.time()
    #     # print(20 * '**')
    #     # print('mydict:', mydict[i])
    #     dist_X_weight = []
    #     for j in range(len(bn['V'])):
    #         # print(20 * '---')
    #         X = bn['V'][j]
    #         # print('current variable:', X)
    #         X_pares = bn['parents'][X] + [X]  # X_pares denotes 'pares + X'
    #         # print(X_pares)
    #         X_cpds = bn['learn_cpds_df'][X]
    #         # print(X_cpds)
    #         t1_1 = time.time()
    #         if not X_pares:  # if current variable has no parents
    #             X_cpds = X_cpds.loc[X_cpds[X] == mydict[i][X], 'p']
    #             # print('x_cpds:\n', X_cpds.values[0])
    #             dist_X_weight.append(X_cpds.values[0])
    #             # print('dist_X_weight:', dist_X_weight)
    #         else:
    #             # temp_X_cpds = pd.DataFrame()
    #             for cur_X in X_pares:
    #                 # print(50*'***')
    #                 # print('cur_X:', cur_X)
    #                 # print('x_cpds:\n', X_cpds)
    #                 # print('my_dict[i]:\n', mydict[i])
    #                 X_cpds = X_cpds.loc[X_cpds[cur_X] == mydict[i][cur_X]]
    #                 # print('retrive_X_cpds:\n', X_cpds)
    #             dist_X_weight.append(X_cpds.loc[:, 'p'].values[0])
    #             # print('dist_X_weight:', dist_X_weight)
    #         # t1_2 = time.time()
    #         # print("time1-2-1=", "{:.4f}".format(t1_2 - t1_1))
    #         # exit(0)
    #     # print('dist_X_weight:', dist_X_weight)
    #     dist_X_weight_product = np.prod(dist_X_weight)
    #     # print('dist_X_weight_product:', dist_X_weight_product)
    #     new_weights.append(dist_X_weight_product)

        # t2 = time.time()
        # print("time1-2=", "{:.4f}".format(t2 - t1))
    new_new_df['wts'] = new_weights
    # print(new_new_df)
    # exit(0)
    return new_new_df


# Maximisation Step
def Maximisation(df, net, mis_index):
    """
    Updates the CPTs of all the nodes based on data given weight of each data point
    Input:
        df - new_df represents fractional samples without weights
        net - Bayesian Net
        Weights - weight corresponding to each fractional sample
            df:
                        "Hypovolemia"  "StrokeVolume"  "LVFailure"  ...   "VentLung"  "Intubation"           wts
                0                1.0             1.0          1.0   ...        2.0           0.0          5.794086e-07
                0                1.0             1.0          1.0   ...        2.0           0.0          2.409856e-06
                0                1.0             1.0          1.0   ...        2.0           0.0          9.999954e-01
                0                1.0             1.0          1.0   ...        1.0           2.0          1.0
                .                .               .            .                .             .            .
    Output:
        "shunt" cpt_data:
                 "Shunt"  "PulmEmbolus"  "Intubation"       counts         p
            0         0              0             0     9.304624  0.097087
            1         1              0             0   100.857966  0.902913
            2         0              0             1     0.113847  0.200000
            3         1              0             1     3.000003  0.800000
            4         0              0             2     0.005804  0.111111
            5         1              0             2     7.033253  0.888889
            6         0              1             0  9601.819160  0.951130
            7         1              1             0   493.706823  0.048870
            8         0              1             1   302.140667  0.955479
            9         1              1             1    12.412881  0.044521
            10        0              1             2    25.197979  0.045028
            11        1              1             2   544.406992  0.954972
    """

    # print('df:\n', df)
    for var in net['V']:
        parents = net['parents'][var]
        var_card = net['cardinality'][var]  # get the cardinality of var
        n_pares = len(parents)
        if n_pares == 0:  # var has not parents
            # print('var:', var)
            counts = []
            for p0 in range(0, var_card):
                df_p0 = df[df[var] == p0]  # retrieve the entries df[var]==p0 of df
                count = np.round(float(df_p0['wts'].sum()), 4)  # accumulate the 'wts' on condition that df[var]==p0
                counts.append(count)
            # if mis_index
            net['learn_cpds_df'][var]['count'] = counts  # learn_cpds_df have been added 'count' column
            # print(net['learn_cpds_df'][var])
            my_bayesnet.normalise_cpt(net, var)  # normalize the cpt again
            # print(net['learn_cpds_df'][var])


        elif n_pares > 0:  # var has parents
            # print('var:', var)
            var_pares = [var] + parents
            wts_sum = np.round(df.groupby(var_pares)['wts'].sum(), 4)
            # print('wts_sum:', var_pares, '\n', wts_sum.tolist())
            """
            wts_sum.index.tolist(): [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0),(0.0, 1.0, 0.0),...]
            """
            bn_count = pd.DataFrame(wts_sum.index.tolist(), columns=var_pares)
            bn_count['count'] = wts_sum.tolist()
            # the columns of 'learn_cpds_df' include 'var pare p count', here we only need columns 'var pare p'
            net['learn_cpds_df'][var] = pd.merge(net['learn_cpds_df'][var].loc[:, var_pares + ['p']],
                                                 bn_count, on=var_pares, how='outer')
            # print(net['learn_cpds_df'][var])
            my_bayesnet.normalise_cpt(net, var)
            # print(net['learn_cpds_df'][var])
            # print('net:\n', net)
    # exit(0)


# Expectation-Maximisation
def Expectation_Maximisation(df, bn, mis_index, output_file):
    """
    Input:
        df - Data Table
        bn - Bayesian Network
        mis_index - array of missing indices corresponding to the Data table 'df'
    Output:
        bn - Bayesian Net with complete parameters learned from the given data by EM algorithm
    """
    curr_iter = 1
    time_i = time.time()
    while True:
        print("ITERATION #" + str(curr_iter))
        t_start = time.time()
        prev_cpts = []  # pre_cpts denotes the initialized cpt by init_param method
        for X in bn['V']:
            prev_cpts.append(np.array(list(bn['learn_cpds_df'][X]['p'])))
            # print(bn['learn_cpds_df'][X])
            # print(bn['learn_cpds_df'][X]['p'])
            # print(prev_cpts)
        # print('prev_cpts:\n', prev_cpts)
        # exit(0)

        step0 = time.time()
        # print "STEP E: "+ str(curr_iter)
        new_df = Expectation(bn, df, mis_index) # 填充后数据
        # print(df)
        # print(new_df)
        # exit(0)

        step1 = time.time()
        # print "STEP M: "+ str(curr_iter)
        Maximisation(new_df, bn, mis_index)
        # print(new_df)
        # exit(0)

        step2 = time.time()
        # print "E time: (%ss)" % (round((step1 - step0), 5))
        # print "M time: (%ss)" % (round((step2 - step1), 5))
        new_cpts = []  # new_cpts denotes the learned cpt by EM algorithm
        for X in bn['V']:
            new_cpts.append(np.array(list(bn['learn_cpds_df'][X]['p'])))
            # print(bn['learn_cpds_df'][X])
            # print(bn['learn_cpds_df'][X]['p'])
            # print(new_cpts)
        # print('new_cpts:\n', new_cpts)
        # exit(0)
        diffs = []
        for i in range(len(prev_cpts)):
            max_diff = max(abs(np.subtract(prev_cpts[i], new_cpts[i])))  # subtract denotes the - operation
            diffs.append(max_diff)
        # print(diffs)
        delta = np.round(max(diffs), 4)
        # print(delta)
        # exit(0)
        time_f = time.time()
        print("Delta: " + str(delta))
        t_stop = time.time()
        print("ITERATION Time: " + str(round((t_stop - t_start), 4)))
        # if (time_f - time_i) > 660:
        #     print("OVER TIME. . . . ")
        #     break

        BIF = BIFWriter(bn)
        BIF.write_bif(output_file + '_EM_' + str(curr_iter) + '.bif')

        if delta <= para_init.threshold_delta or curr_iter == para_init.max_iter:
            break
        curr_iter += 1
    print("Converged in (" + str(curr_iter) + ") iterations")



    return bn


if __name__ == '__main__':
    # #%%
    # start_time = time.time()
    # bn, df, mis_index = setup_network('data/bn/alarm.bif', 'data/missingdata/500/alarm_500_TPR_CO.xlsx')
    # # bn, df, mis_index = setup_network('data/bn/link.bif', 'data/missingdata/500/link_500_N73.xlsx')
    # bn, df, mis_index = setup_network('data/bn/child.bif', 'data/missingdata/500/4/child_500_disease_LungParench_Sick_Age.xlsx')
    # # # print('mis_index:', mis_index)
    # # # print('bn:\n', bn)
    # # #%%
    # # # writer = BIFWriter(bn)
    # # # writer.write_bif('data/network_initparam_written.bif')
    # network = Expectation_Maximisation(df, bn, mis_index)
    # end_time = time.time()
    # print('runtime:', end_time - start_time)
    # # # print('network:\n', network)
    #%%
    # writer = BIFWriter(network)
    # writer.write_bif('data/network_EM_written.bif')

    # bn_name = ['child', 'alarm', 'hepar2', 'andes', 'munin1', 'link']
    # bn_name = ['child']
    # bn_data = ['data/bn/child.bif', 'data/bn/alarm.bif', 'data/bn/hepar2.bif',
    #            'data/bn/andes.bif', 'data/bn/munin1.bif', 'data/bn/link.bif']
    # bn_name = ['child']
    # bn_data = ['data/bn/remove_edges_rename_child.bif']

    bn_name = [para_init.bn]
    bn_data = para_init.bn_data
    missingdata = para_init.missingdata
    output_file = para_init.output_file

    # # two latent variables
    # missingdata = ['data/missingdata/500/child_500_disease_LungParench.xlsx',
    #                'data/missingdata/500/alarm_500_TPR_CO.xlsx',
    #                'data/missingdata/500/hepar2_500_surgery_fat.xlsx',
    #                'data/missingdata/500/andes_500_SNode_3_SNode_4.xlsx',
    #                'data/missingdata/500/munin1_500_MUDENS_MALOSS.xlsx',
    #                'data/missingdata/500/link_500_N73.xlsx']

    # one latent variables
    # CHILD: P(Disease:1 | BirthAsphyxia:0)
    # ALRAM: P(TPR:0 | ANAPHYLAXIS:0)
    # HEAPR2: P(fat:1 | gallstones:0)
    # ANDES: P(SNode_3:1)
    # MUNIN1: P(R_MYAS_APB_MUDENS:0)
    # LINK: P(D0_33_d_p:0 | N33_d_g:0)
    # missingdata = ['data/missingdata/100/1/child_100_disease.xlsx',
    #                'data/missingdata/100/1/alarm_100_TPR.xlsx',
    #                'data/missingdata/100/1/hepar2_100_fat.xlsx',
    #                'data/missingdata/100/1/andes_100_SNODE_3.xlsx',
    #                'data/missingdata/100/1/munin1_100_MUDENS.xlsx',
    #                'data/missingdata/100/1/link_100_D0_33_d_p.xlsx']
    # mis_var_value = [{'Disease': 1}, {'TPR': 0}, {'fat': 1}, {'SNode_3': 1}, {'R_MYAS_APB_MUDENS': 0}, {'D0_33_d_p': 0}]


    # two latent variables
    # missingdata = ['data/missingdata/100/2/child_100_disease_LungParench.xlsx',
    #                'data/missingdata/100/2/alarm_100_TPR_CO.xlsx',
    #                'data/missingdata/100/2/hepar2_100_surgery_fat.xlsx',
    #                'data/missingdata/100/2/andes_100_SNode_3_SNode_4.xlsx',
    #                'data/missingdata/100/2/munin1_100_MUDENS_MALOSS.xlsx',
    #                'data/missingdata/100/2/link_100_N73.xlsx']

    # three latent variables
    # missingdata = ['data/missingdata/100/3/child.xlsx',
    #                'data/missingdata/100/3/alarm.xlsx',
    #                'data/missingdata/100/3/hepar2.xlsx',
    #                'data/missingdata/100/3/andes.xlsx',
    #                'data/missingdata/100/3/munin1.xlsx',
    #                'data/missingdata/100/3/link.xlsx']

    # four latent variables
    # missingdata = ['data/missingdata/100/4/child_100_disease_LungParench_Sick_Age.xlsx',
    #                'data/missingdata/100/4/alarm_100_TPR_CO_PAP_FI02.xlsx',
    #                'data/missingdata/100/4/hepar2_100_surgery_fat_age_sex.xlsx',
    #                'data/missingdata/100/4/andes_100_SNode_3_SNode_4_STRAT_90_SNode_13.xlsx',
    #                'data/missingdata/100/4/munin1_100_MUDENS_MALOSS_R_LNLLP_APB_MUSIZE.xlsx',
    #                'data/missingdata/100/4/link_100_N73_N6_a_m_N6_a_f.xlsx']
    # missingdata = ['data/missingdata/1000/4/child_1000.xlsx']
    # missingdata = ['data/my_toy/toy2_origin_12.xlsx']
    # missingdata = ['data/my_toy/toy2_origin_L2_12.xlsx']


    # five latent variables
    # missingdata = ['data/missingdata/100/5/child.xlsx',
    #                'data/missingdata/100/5/alarm.xlsx',
    #                'data/missingdata/100/5/hepar2.xlsx',
    #                'data/missingdata/100/5/andes.xlsx',
    #                'data/missingdata/100/5/munin1.xlsx',
    #                'data/missingdata/100/5/link.xlsx']

    # six latent variables
    # missingdata = ['data/missingdata/100/6/child_100.xlsx',
    #                'data/missingdata/100/6/alarm_100.xlsx',
    #                'data/missingdata/100/6/hepar2_100.xlsx',
    #                'data/missingdata/100/6/andes_100.xlsx',
    #                'data/missingdata/100/6/munin1_100.xlsx',
    #                'data/missingdata/100/6/link_100.xlsx']

    # for i in range(6):
    for i in range(len(bn_name)):
        print(30 * "***")
        print('bn_name:', bn_name[i])
        start_time = time.time()
        bn, df, mis_index = setup_network(bn_data[i], missingdata[i])
        # print(mis_index)
        # exit(0)

        # ori_cpts = []
        # for X in bn['V']:
        #     ori_cpts.append(list(bn['learn_cpds_df'][X]['p']))
        # ori_cpts = [i for j in ori_cpts for i in j]

        # mis_var = df.columns.tolist()[mis_index[0]]
        # parents = bn['parents'][mis_var]
        # print('parents:', parents)
        # mis_var_value_store = mis_var_value[i]
        # print('mis_var_value_store:', mis_var_value_store)
        # if not parents:
        #     X_pares_value_store = {}
        # else:
        #     X_pares_value_store = {parents[0]: 0}
        # # print('X_pares_value_store:', X_pares_value_store)
        # ori_mis_par_p = my_bayesnet.retrieve_p(bn, mis_var, X_pares_value_store, mis_var_value_store)
        # print('ori_mis_par_p:\n', ori_mis_par_p)


        # for var in bn['V']:
        #     flag = 0
        #     parents_var = bn['parents'][var]
        #     parents_var.append(var)
        #     # print(parents_var)
        #     # print(mis_index)
        #     for j in mis_index:
        #         X = df.columns.tolist()[j]
        #         # print('L: ' + str(X))
        #         if str(X) in parents_var:
        #             flag = 1
        #             break
        #     if flag == 1:
        #         print(var)
        # exit(0)

        network = Expectation_Maximisation(df, bn, mis_index, output_file[i])

        end_time = time.time()
        print('runtime:', end_time - start_time)

        # cur_cpts = []
        # for X in network['V']:
        #     cur_cpts.append(list(network['learn_cpds_df'][X]['p']))
        # cur_cpts = [i for j in cur_cpts for i in j]
        # print('MSE:', MSE(np.array(cur_cpts), np.array(ori_cpts)))
        # print('KL:', KL(ori_cpts, cur_cpts))

        # cur_mis_par_p = my_bayesnet.retrieve_p(network, mis_var, X_pares_value_store, mis_var_value_store)
        # print('cur_mis_par_p:', cur_mis_par_p)

        # BIF = BIFWriter(network)
        # BIF.write_bif(output_file[i] + '_EM.bif')







