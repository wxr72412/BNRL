from __future__ import division
from BN.my_BIF import BIFReader, BIFWriter
from BN import my_bayesnet
import numpy as np
import pandas as pd
import time
from metrics import MSE, KL
import para_init

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 显示行的长度
pd.set_option('display.width', 1000)


# Setup the network
def setup_network(net_bif, dat_records):
    # Parsing the network from .bif format
    print("0: Reading Network . . . ")
    reader = BIFReader(net_bif)
    net = reader.my_model()
    # Get data from record.dat
    print("1: Getting data from records . . . ")
    df = my_bayesnet.read_excel_data(dat_records)

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


# get the row of the cpds satisfying var=value, mb = mb_value
def get_assignment_for(cpds, mb, var, value):
    """
    Input:
        cpds: cpds represents the cpt_data of variable having missing value， factor is stored as dataframe format
        mb stores the markov blanket variables of latent variable and mb variables' values
        mb like: {'"PAP"': 1.0, '"Shunt"': 0.0, '"Intubation"': 0.0}
        nval: nval represents the number of values of variable having missing value
    Output:
        learn_cpds: return the rows of the cpds that the assignment of parents of missing value variable as specified
                    in E=MB(X)
             "VentTube"  "VentMach"  "Disconnect"  counts         p
        20           0           2             1      97  0.010524
    """
    learn_cpds = cpds  # learn_cpds=cpds, cpds is dataframe including p and counts
    mb[var] = value  # add current var and its value into the mb
    for key, value in mb.items():  # mb like: {'"PAP"': 1.0, '"Shunt"': 0.0, '"Intubation"': 0.0}
        if key in list(cpds.columns):  # traverse the assigned items that the parents are same
            condition = learn_cpds[key] == value
            # learn_cpds is reassigned value, filter cpt value being similar to the value of MB
            learn_cpds = learn_cpds[condition]
            # print('learn_cpds:\n', learn_cpds)
        # if learn_cpds.shape[0] == 1:
        #     return learn_cpds

    return learn_cpds


# Inference by Markov Blanket Sampling
def markov_blanket_sampling(mis_var, mydict, mb, bn):
    """
    Input:
        var: mis_index, represents the missing value variable
        mb: markov blanket of mis_index
        bn: bayesian network
    Output:
        fac_c: represents the sum of probability of MB(x), like p(x|MB(x))=log p(x|pa(x))+log p(child(x)|x)
        [-1.62525305 -0.23173922]=[-1.60463376 -0.2243482 ]+[-0.02061929 -0.00739102]
    """
    x_cpt = bn['learn_cpds_df'][mis_var]
    # print('x_cpt:\n', x_cpt)
    children = bn['children'][mis_var]
    # print(children)

    # print(20 * '**')
    # fac_x gets the entry of cpt satisfying mis_var == value and mb == value
    fac_x = get_assignment_for(x_cpt, mb, mis_var, mydict[mis_var])
    # print('fac_x:\n', fac_x)
    # fac_c = np.log(np.asarray(fac_x['p']))  # fac_c like: [-0.055249   -3.09385426 -4.77671965]
    fac_c = np.asarray(fac_x['p'])
    # print('fac_c:\n', fac_c)
    for c in children:
        c_cpt = bn['learn_cpds_df'][c]
        # print('c_cpt:', c_cpt)
        temp = get_assignment_for(c_cpt, mb, c, mydict[c])
        # print('c_temp:\n', temp)
        fac_c = fac_c * np.asarray(temp['p'])
        # print('fac_c:', fac_c)

    # print('fac_c:\n', fac_c)
    # exit(0)
    return fac_c


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
    new_new_df = pd.DataFrame()
    # new_df_list = []
    for i in range(df.shape[0]):  # i represents the i th row
        # t1 = time.time()
        # print(20*'***')
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
        row = pd.DataFrame(df.loc[i, :]).T
        # print('row:\n', row)
        # mis_index = [1, 5],1 and 5 represents having two the latent variables, their index being 1 and 5 respectively
        for j in mis_index:
            X = df.columns.tolist()[j]
            # X = bn['V'][mis_index[j]]  # the name of variable having missing value
            if len(new_df.index) == 0:
                for n in range(bn['cardinality'][X]):
                    row.iloc[0, j] = n  # the value of X in row is replaced by current n
                    new_df = pd.concat([new_df, row])  # concat the row (df type) to new_df
                new_df = new_df.reset_index(drop=True)
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
        # t2 = time.time()
        # print("time1-1=", "{:.4f}".format(t2 - t1))
    # print(new_new_df)
    # exit(0)
    new_weights = []
    # dict(orient="records") like [{'"LVEDVolume"': 0, '"Hypovolemia"': 1...}, {'"LVEDVolume"': 22, ...}]
    mydict = new_new_df.to_dict(orient='records')
    # new_df_list = []
    for i in range(new_new_df.shape[0]):  # i represents the row of df
        # print(i)
        t1 = time.time()
        # print(20 * '**')
        # print('mydict:', mydict[i])
        dist_X_weight = []
        for j in mis_index:
            t1_1 = time.time()
            # print('j:', j)
            X = df.columns.tolist()[j]
            # print('current variable:', X)
            X_value = mydict[i][X]
            mb_x = bn['mb'][X]
            # print('mb_x:', mb_x)
            # E stores the markov blanket variables and its values
            mb = {key: value for key, value in mydict[i].items() if (key != X and key in mb_x)}
            # print(mb)
            # print('E:\n', E)
            dist_X = markov_blanket_sampling(X, mydict[i], mb, bn)
            # print(dist_X)
            dist_X_weight.append(dist_X)
            t1_2 = time.time()
            # print("time1-2-1=", "{:.4f}".format(t1_2 - t1_1))
        # print(dist_X_weight)
        dist_X_sum = np.sum(dist_X_weight)
        # print(dist_X_sum)
        new_weights.append(dist_X_sum)
        t2 = time.time()
        # print("time1-2=", "{:.4f}".format(t2 - t1))
        # exit(0)
    new_new_df['wts'] = new_weights
    # print(new_new_df)
    # exit(0)
    return new_new_df


# Maximisation Step
def Maximisation(df, net):
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
            counts = []
            for p0 in range(0, var_card):
                df_p0 = df[df[var] == p0]  # retrieve the entries df[var]==p0 of df
                count = float(df_p0['wts'].sum())  # accumulate the 'wts' on condition that df[var]==p0
                counts.append(count)
                # if count != 0:
                #     counts.append(float(count))
                # else:
                #     counts.append(np.nan)
            net['learn_cpds_df'][var]['count'] = counts  # learn_cpds_df have been added 'count' column
            my_bayesnet.normalise_cpt(net, var)  # normalize the cpt again

        elif n_pares > 0:  # var has parents
            var_pares = [var] + parents
            wts_sum = df.groupby(var_pares)['wts'].sum()
            # print('wts_sum:', var_pares, '\n', wts_sum.tolist())
            """
            wts_sum.index.tolist(): [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0),(0.0, 1.0, 0.0),...]
            """
            bn_count = pd.DataFrame(wts_sum.index.tolist(), columns=var_pares)
            bn_count['count'] = wts_sum.tolist()
            # print('bn_count:\n', bn_count)
            # the columns of 'learn_cpds_df' include 'var pare p count', here we only need columns 'var pare p'
            net['learn_cpds_df'][var] = pd.merge(net['learn_cpds_df'][var].loc[:, var_pares + ['p']],
                                                 bn_count, on=var_pares, how='outer')
            # print('\nlearn_cpds_df:' + var + '\n', net['learn_cpds_df'][var])
            my_bayesnet.normalise_cpt(net, var)


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
        # print('prev_cpts:\n', prev_cpts)
        # step0 = time.time()
        # print "STEP E: "+ str(curr_iter)
        new_df = Expectation(bn, df, mis_index)
        step1 = time.time()
        # print "STEP M: "+ str(curr_iter)
        Maximisation(new_df, bn)
        # step2 = time.time()
        # print "E time: (%ss)" % (round((step1 - step0), 5))
        # print "M time: (%ss)" % (round((step2 - step1), 5))
        new_cpts = []  # new_cpts denotes the learned cpt by EM algorithm
        for X in bn['V']:
            new_cpts.append(np.array(list(bn['learn_cpds_df'][X]['p'])))
        # print('new_cpts:\n', new_cpts)
        diffs = []
        for i in range(len(prev_cpts)):
            max_diff = max(abs(np.subtract(prev_cpts[i], new_cpts[i])))  # subtract denotes the - operation
            diffs.append(max_diff)
        delta = np.round(max(diffs), 4)
        time_f = time.time()
        print("Delta: " + str(delta))
        t_stop = time.time()
        print("ITERATION Time: " + str(round((t_stop - t_start), 4)))
        # if (time_f - time_i) > 660:
        #     print("OVER TIME. . . . ")
        #     break

        BIF = BIFWriter(bn)
        BIF.write_bif(output_file + '_IEM_' + str(curr_iter) + '.bif')

        if delta <= para_init.threshold_delta or curr_iter == para_init.max_iter:
        # if delta <= 0.06:
            break
        curr_iter += 1
    print("Converged in (" + str(curr_iter) + ") iterations")

    return bn


if __name__ == '__main__':
    # #%%
    # bn, df, mis_index = setup_network('data/bn/child.bif', 'data/missingdata/100/1/child_100_disease.xlsx')
    # bn, df, mis_index = setup_network('data/bn/link.bif', 'data/missingdata/100/1/link_100_D0_33_d_p.xlsx')
    # # print('bn:\n', bn)
    # mis_var = df.columns.tolist()[mis_index[0]]
    # parents = bn['parents'][mis_var]
    # print('parents:', parents)
    # mis_var_value_store = {mis_var: 0}
    # print('mis_var_value_store:', mis_var_value_store)
    # X_pares_value_store = {parents[0]: 0}
    # print('X_pares_value_store:', X_pares_value_store)
    # ori_Disease_p = bn['learn_cpds_df'][mis_var]

    # mis_par_p = my_bayesnet.retrieve_p(bn, mis_var, mis_var_value_store, X_pares_value_store)


    # #%%
    # writer = BIFWriter(bn)
    # writer.write_bif('data/network_initparam_written.bif')
    # network = Expectation_Maximisation(df, bn, mis_index)
    # # print('network:\n', network)
    # #%%
    # writer = BIFWriter(network)
    # writer.write_bif('data/network_EM_written.bif')

    # bn_name = ['child', 'alarm', 'hepar2', 'andes', 'munin1', 'link']
    # bn_data = ['data/bn/child.bif', 'data/bn/alarm.bif', 'data/bn/hepar2.bif',
    #            'data/bn/andes.bif', 'data/bn/munin1.bif', 'data/bn/link.bif']

    bn_name = [para_init.bn]
    bn_data = para_init.bn_data
    missingdata = para_init.missingdata
    output_file = para_init.output_file

    # # 500 two latent variables
    # missingdata = ['data/missingdata/500/child_500_disease_LungParench.xlsx', 'data/missingdata/500/alarm_500_TPR_CO.xlsx',
    #                'data/missingdata/500/hepar2_500_surgery_fat.xlsx', 'data/missingdata/500/andes_500_SNode_3_SNode_4.xlsx',
    #                'data/missingdata/500/munin1_500_MUDENS_MALOSS.xlsx', 'data/missingdata/500/link_500_N73.xlsx']

    # one latent variable
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
    # missingdata = ['data/my_toy/toy2_origin_L2_12.xlsx']

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

    for i in range(len(bn_name)):
        print(30 * "***")
        print('bn_name:', bn_name[i])
        start_time = time.time()
        bn, df, mis_index = setup_network(bn_data[i], missingdata[i])

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
        # ori_cpts = []
        # for X in bn['V']:
        #     ori_cpts.append(list(bn['learn_cpds_df'][X]['p']))
        # ori_cpts = [i for j in ori_cpts for i in j]



        network = Expectation_Maximisation(df, bn, mis_index, output_file[i])
        end_time = time.time()
        print('runtime:', end_time - start_time)
        # cur_mis_par_p = my_bayesnet.retrieve_p(network, mis_var, X_pares_value_store, mis_var_value_store)
        # print('cur_mis_par_p:', cur_mis_par_p)

        # cur_cpts = []
        # for X in network['V']:
        #     cur_cpts.append(list(network['learn_cpds_df'][X]['p']))
        # cur_cpts = [i for j in cur_cpts for i in j]
        # print('MSE:', MSE(np.array(cur_cpts), np.array(ori_cpts)))
        # print('KL:', KL(ori_cpts, cur_cpts))

        # BIF = BIFWriter(network)
        # BIF.write_bif(output_file[i] + '_IEM.bif')

    # bn, df, mis_index = setup_network('data/network.bif', 'data/bn_data_1000.xlsx')
    #
    # new_new_df = pd.DataFrame()
    # # new_df_list = []
    # for i in range(df.shape[0]):  # i represents the i th row
    #     # print(20*'***')
    #     """
    #     row = pd.DataFrame(df.loc[i, :]).T:
    #
    #                   size  neighborhood  children  schools  amenities  location  age  price
    #             0       1             0         0        1        NaN        NaN  NaN    2.0
    #             1       2             1         0        0        NaN        NaN  NaN    0.0
    #             2       2             1         1        1        NaN        NaN  NaN    1.0
    #             3       0             1         1        0        NaN        NaN  NaN    1.0
    #             4       1             0         0        1        NaN        NaN  NaN    1.0
    #     """
    #     new_df = pd.DataFrame()
    #     row = pd.DataFrame(df.loc[i, :]).T
    #     # print('row:\n', row)
    #     # mis_index = [1, 5],1 and 5 represents having two the latent variables, their index being 1 and 5 respectively
    #     for j in range(len(mis_index)):
    #         X = bn['V'][mis_index[j]]  # the name of variable having missing value
    #         if len(new_df.index) == 0:
    #             for n in range(bn['cardinality'][X]):
    #                 row.iloc[0, bn['V'].index(X)] = n  # the value of X in row is replaced by current n
    #                 new_df = pd.concat([new_df, row])  # concat the row (df type) to new_df
    #             new_df = new_df.reset_index(drop=True)
    #         else:
    #             for r in range(new_df.shape[0]):
    #                 new_row = pd.DataFrame(new_df.loc[0, :]).T  # acquire the row of new_df
    #                 # print('new_row.loc[0, {j}]:{v}'.format(j=mis_index[j], v=new_row.iloc[0, mis_index[j]]))
    #                 if np.isnan(new_row.iloc[0, mis_index[j]]):  # judge the new_row whether the mis_index[j] has NaN
    #                     # if the new_row has NaN element, traverse the cardinality of X
    #                     for n in range(bn['cardinality'][X]):
    #                         new_row.iloc[0, bn['V'].index(X)] = n
    #                         new_df = pd.concat([new_df, new_row])
    #                         # reindex the row of new_df in order to drop the new_df.loc[0, :] conveniently
    #                         new_df = new_df.reset_index(drop=True)
    #                     new_df.drop([0], axis=0, inplace=True)  # delete the 1st row in the new_df
    #                     # reindex the row of new_df in order to find the new row (new_df.loc[0, :]) conveniently
    #                     new_df = new_df.reset_index(drop=True)
    #                     # print('new_df:/n', new_df)
    #                 else:
    #                     break
    #     new_new_df = pd.concat([new_new_df, new_df])
    #     new_new_df = new_new_df.reset_index(drop=True)
    # print('new_new_df:\n', new_new_df)
    # new_weights = []
    # # dict(orient="records") like [{'"LVEDVolume"': 0, '"Hypovolemia"': 1...}, {'"LVEDVolume"': 22, ...}]
    # mydict = new_new_df.to_dict(orient='records')
    # # new_df_list = []
    # for i in range(new_new_df.shape[0]):  # i represents the row of df
    #     print(20 * '***')
    #     print('mydict:', mydict[i])
    #     dist_X_weight = []
    #     for j in mis_index:
    #         X = bn['V'][j]
    #         X_value = mydict[i][X]
    #         print(20 * '---')
    #         print('{}:{}'.format(X, X_value))
    #         mb_x = bn['mb'][X]
    #         # E stores the markov blanket variables and its values
    #         mb = {key: value for key, value in mydict[i].items() if (key != X and key in mb_x)}
    #         print('mb:\n', mb)
    #         dist_X = markov_blanket_sampling(X, mydict[i], mb, bn)
    #         print('dist_X:', dist_X)
    #         dist_X_weight.append(dist_X)
    #         print('dist_X_weight:', dist_X_weight)
    #     dist_X_sum = np.sum(dist_X_weight)
    #     print('dist_X_sum:', dist_X_sum)
    #     new_weights.append(dist_X_sum)
    #     print('new_weights:', new_weights)
    # new_new_df['wts'] = new_weights
    #
    # print('new_new_df:\n', new_new_df)

