from __future__ import division
import numpy as np
import pandas as pd
import time
from scipy.sparse.linalg import svds
from copy import deepcopy

from BN.my_BIF import BIFReader
from DE import incremental_SVD
from BN import BNSampling, my_bayesnet
from BN.my_bayesnet import save_excel
from metrics import MSE, KL
import para_init

# show all columns
pd.set_option('display.max_columns', None)
# show all rows
pd.set_option('display.max_rows', None)
# show the length of the row
pd.set_option('display.width', 1000)


def get_learn_cpd(bn):
    """
    Returns
    -------
    dict: dict of type {variable: array}

    Example
    -------
    bn, df, mis_index = setup_network('data/network.bif', 'data/bn_data_1000.xlsx')
    bn/network = Expectation_Maximisation(df, bn, mis_index)
    learnt_cpds:
    {'bowel-problem': array([ 0.01,  0.99]),
     'dog-out': array([ 0.99,  0.97,  0.9 ,  0.3 ,  0.01,  0.03,  0.1 ,  0.7 ]),
     'family-out': array([ 0.15,  0.85]),
     'hear-bark': array([ 0.7 ,  0.01,  0.3 ,  0.99]),
     'light-on': array([ 0.6 ,  0.05,  0.4 ,  0.95])}
    """
    cpds = bn['learn_cpds_df']
    variables = bn['V']
    tables = {}
    for var in variables:
        cpd_value = np.array(cpds[var]['p'])  # get the probability of var
        # print('cpd_value:\n', cpd_value)
        # print('type:', np.array(cpd_value))
        # cpd_value = np.array(cpd_value)
        pares = bn['parents'][var]
        if pares:
            cpd_value = cpd_value.reshape(cpd_value.size // bn['cardinality'][var],
                                          bn['cardinality'][var])  # reshape the np.array
        tables[var] = cpd_value
        # print('tables:', tables)
        bn['learn_cpds'] = tables
    return bn


# Setup the network
def setup_network(net_bif, dat_records):
    # Parsing the network from .bif format
    print("0: Reading Network . . . ")
    reader = BIFReader(net_bif)
    net = reader.my_model()
    # Get data from record.dat
    print("1: Getting data from records . . . ")
    df = my_bayesnet.read_excel_data(dat_records)
    # Initialise parameters
    print("2: Initialising parameters . . . ")
    net = my_bayesnet.my_init_param(df, net)
    # net.setdefault('learn_cpds_df_wts', deepcopy(net['learn_cpds_df']))
    # Get the index of nodes which have missing value in each row
    print("3: Getting missing data indexes . . . ")
    mis_index = get_missing_index(df)
    return net, df, mis_index


# List of the indexes of latent variables, such as mis_index = [3, 4], representing the index of the latent variable;
def get_missing_index(df):  # get the index of column of df (dataset)
    """
        param df: df is read from record.dat
                       "Hypovolemia"  "StrokeVolume"  ...  "VentLung"  "Intubation"
            0                1.0             1.0  ...         NaN           NaN
            1                1.0             1.0  ...         NaN           NaN
            2                1.0             1.0  ...         NaN           NaN
            3                0.0             0.0  ...         NaN           NaN
            4                0.0             0.0  ...         NaN           NaN
        return e.g., [3,4] represents the index of the latent variable
            [10, 2, 30, 4, 26, 21, 30, 17, 21, 3, 25, 14, 12, 18, 32, 26, 9, 34, 30, 29, ....]
        """
    row_mis = df.iloc[0, :].T
    # return the index of nonzero, e.g., array([[4], [5], [6]])
    # print('np.isnan:\n', np.isnan(np.asarray(row_mis)))
    mis_index = np.argwhere(np.isnan(np.asarray(row_mis)))
    mis_index = [int(i) for i in mis_index]
    return mis_index


# Normalise a numpy array
def normalise_array(bn, mis_var, X_pares_value_store, mis_var_value_store, sim_mis_var):
    """
    input:
        bn, mis_var,
        X_pares_value_store: {'pare0':0, 'pare0':1, 'pare1':0, ...}
        Sim_mis_var: Sim(x|neighbor(x))=Sim(pa(x)->X)+Sim(x->child(x)), e.g. [-1.62525305 -0.23173922]
    output:
        normalised_vals.append(val / float(denom))

    """
    # print('mis_var: {k} {v}'.format(k=mis_var, v=mis_var_value_store[mis_var]))
    if sim_mis_var == 0:  # if sim_mis_var is zero
        mis_var_pare_df = bn['learn_cpds_df'][mis_var]
        """
        X_pares_value_store such as {'pare0':0, 'pare0':1, 'pare1':0, ...}
        """
        # print('X_pares_value_store:', X_pares_value_store)
        # if the mis_var has no parents
        if not X_pares_value_store:
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
            # for k, v in X_pares_value_store.items():
            #     temp_df = mis_var_pare_df.loc[mis_var_pare_df[k] == v]
            #     # reset the index of temp_df in order to the value of mis_var, e.g.,
            #     # mis_var_pare_df[mis_var_value_store[mis_var]]
            #     mis_var_pare_df = temp_df['p'].reset_index(drop=True)
            # temp_value = mis_var_pare_df[mis_var_value_store[mis_var]]
            # # print('temp_value:', temp_value)
    else:
        temp_value = sim_mis_var
    return temp_value


def neighbor_similarity(bn, mis_var, mis_var_value, mis_var_value_store, U_cur, V_cur, PMI_IJK_index_name,
                        X_pares_value, X_childs_value, X_pares_value_store):
    """
    Input:
        var: mis_index, represents the missing value variable
        bn: bayesian network
    Output:
        similarity: represents the sum of similarity, like Sim(x|neighbor(x))=Sim(pa(x)->X)+Sim(x->child(x))
        [1.62525305 0.23173922]=[1.60463376 0.2243482 ]+[0.02061929 0.00739102]
        the similarity of X w.r.t. its states:   [1.62525305 0.23173922]
                                                    X_state_0   X_state_0
    """

    if X_pares_value and not X_childs_value:
        Sim_temp = []
        # print('X_pares_value:', mis_var_state)
        var_i = PMI_IJK_index_name.index(mis_var_value)  # get the index of '{u}_{s}' in PMI_IJK_index
        for j in range(len(X_pares_value)):
            # get the index of '{u}_{s}' in PMI_IJK_index
            pare_i = PMI_IJK_index_name.index(X_pares_value[j])
            Sim_temp.append(np.dot(U_cur[pare_i], V_cur[var_i].T))
        Sim_mis_var = np.sum(np.array(Sim_temp)) * 10 ** 15  # increase the size of value by multiplying 10^15

    if not X_pares_value and X_childs_value:
        Sim_temp = []
        var_i = PMI_IJK_index_name.index(mis_var_value)
        for j in range(len(X_childs_value)):
            child_i = PMI_IJK_index_name.index(X_childs_value[j])
            Sim_temp.append(np.dot(U_cur[var_i], V_cur[child_i].T))
        Sim_mis_var = np.sum(np.array(Sim_temp)) * 10 ** 15  # increase the size of value by multiplying 10^15
    # print('Sim_mis_var:', Sim_mis_var)

    if X_pares_value and X_childs_value:
        Sim_mis_var_pare = []
        Sim_mis_var_child = []
        var_i = PMI_IJK_index_name.index(mis_var_value)
        for j in range(len(X_pares_value)):
            pare_i = PMI_IJK_index_name.index(X_pares_value[j])
            Sim_mis_var_pare.append(np.dot(U_cur[pare_i], V_cur[var_i].T))
        for j in range(len(X_childs_value)):
            child_i = PMI_IJK_index_name.index(X_childs_value[j])
            Sim_mis_var_child.append(np.dot(U_cur[var_i], V_cur[child_i].T))
        # increase the size of value by multiplying 10^15
        Sim_mis_var = (np.sum(np.array(Sim_mis_var_pare)) + np.sum(np.array(Sim_mis_var_child))) * 10 ** 15
        """
        Sim_mis_var of X w.r.t. its states:   [1.62525305 0.23173922]
                                               X_state_0   X_state_0
        """
    return normalise_array(bn, mis_var, X_pares_value_store, mis_var_value_store, Sim_mis_var)


# Expectation Step
def Expectation(bn, df, mis_index, U_cur, V_cur, PMI_IJK_index_name):
    # print('df:/n', df)
    # new_new_df stores the samples of fractional samples
    new_new_df = pd.DataFrame()
    # new_df_list = []
    for i in range(df.shape[0]):  # i represents the i th row
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
        row = pd.DataFrame(df.loc[i, :]).T
        # print('row:\n', row)
        # mis_index = [1, 5],1 and 5 represents having two the latent variables, their index being 1 and 5 respectively
        for j in mis_index:
            X = df.columns.tolist()[j]  # the name of variable having missing value
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
                        new_df.drop([0], axis=0, inplace=True)
                        # reindex the row of new_df in order to find the new row (new_df.loc[0, :]) conveniently
                        new_df = new_df.reset_index(drop=True)
                        # print('new_df:/n', new_df)
                    else:
                        break
        new_new_df = pd.concat([new_new_df, new_df])
        new_new_df = new_new_df.reset_index(drop=True)

    # new_weights stores the value of fractional sample w.r.t. the one row of new_new_df
    new_weights = []
    # dict(orient="records") like [{'"LVEDVolume"': 0, '"Hypovolemia"': 1...}, {'"LVEDVolume"': 22, ...}]
    mydict = new_new_df.to_dict(orient='records')
    for i in range(new_new_df.shape[0]):
        dist_X_weight = []
        for j in mis_index:
            X = df.columns.tolist()[j]
            X_pares = bn['parents'][X]
            X_childs = bn['children'][X]
            X_value = '{u}_{s}'.format(u=X, s=round(mydict[i][X]))
            X_value_store = {X: int(mydict[i][X])}
            X_pares_value_store = {key: int(value) for key, value in mydict[i].items() if (key != X and key in X_pares)}
            X_pares_value = ['{u}_{s}'.format(u=key, s=int(value)) for key, value in mydict[i].items() if
                             (key != X and key in X_pares)]
            X_childs_value = ['{u}_{s}'.format(u=key, s=int(value)) for key, value in mydict[i].items() if
                              (key != X and key in X_childs)]
            """
            neighbor_similarity:
                represents the sum of similarity, like Sim(x|neighbor(x))=Sim(pa(x)->X)+Sim(x->child(x))
            """
            dist_X = neighbor_similarity(bn, X, X_value, X_value_store, U_cur, V_cur, PMI_IJK_index_name,
                                         X_pares_value, X_childs_value, X_pares_value_store)

            # new_weights.append(dist_X)
            dist_X_weight.append(dist_X)  # append similarity of all the latent variables with its neighbors
        dist_X_weight_sum = np.sum(dist_X_weight)  # sum up all the similarity of the latent variables in the same row
        new_weights.append(dist_X_weight_sum)
    new_new_df['wts'] = new_weights
    # print('new_weights:\n', new_weights)
    # print('new_new_df:\n', new_new_df)
    return new_new_df


# Maximisation Step
def Maximisation(new_df, bn):
    """
        Updates the CPTs of all the nodes based on data given weight of each data point
        Input:
            new_df - new_df represents fractional samples without weights
            bn - Bayesian Network
                new_df:
                            "Hypovolemia"  "StrokeVolume"  "LVFailure"  ...   "VentLung"  "Intubation"           wts
                    0                1.0             1.0          1.0   ...        2.0           0.0          5.794086e-07
                    0                1.0             1.0          1.0   ...        2.0           0.0          2.409856e-06
                    0                1.0             1.0          1.0   ...        2.0           0.0          9.999954e-01
                    0                1.0             1.0          1.0   ...        1.0           2.0          1.0
                    .                .               .            .                .             .            .
        Output:
            "price" cpt_data:
                     price  location  age  schools  size      p     count
                0        0         0    0        0     0  0.663  5.993062
                1        1         0    0        0     0  0.331  2.996531
                2        2         0    0        0     0  0.006  0.050000
                3        0         0    0        0     1  0.016  0.050000
                4        1         0    0        0     1  0.968  2.996531
                5        2         0    0        0     1  0.016  0.050000
                6        0         0    0        0     2  0.459  2.996531
                ......
        """

    for var in bn['V']:
        # print(30 * '==')
        parents = bn['parents'][var]
        var_card = bn['cardinality'][var]  # get the cardinality of var
        n_pares = len(parents)
        if n_pares == 0:  # var has not parents
            counts = []
            for p0 in range(0, var_card):
                df_p0 = new_df[new_df[var] == p0]  # retrieve the entries df[var]==p0 of df
                count = float(df_p0['wts'].sum())  # accumulate the 'wts' on condition that df[var]==p0
                if count != 0:
                    counts.append(float(count))
                else:
                    counts.append(0.05)  # prevent the 'wts' is np.nan
            # print('only_var:' + var + ', only_var counts:' + str(counts))
            bn['learn_cpds_df'][var]['count'] = counts  # learn_cpds_df have been added 'count' column
            # print('\nlearn_cpds_df:' + var + '\n', bn['learn_cpds_df'][var])
            my_bayesnet.normalise_cpt(bn, var)  # normalize the cpt again

        elif n_pares > 0:  # var has parents
            var_pares = [var] + parents
            # print('pares_var:' + str(var_pares))
            """Input:
                wts_sum_group = new_df.groupby(var_pares)
                for name, group in wts_sum_group:
                    print('name:', name)
                    print(group)
                Output:  pares_var:['price', 'location', 'age', 'schools', 'size']

                name: (0.0, 0.0, 0.0, 0.0, 0.0)
                     size  neighborhood  children  schools  amenities  location  age  price       wts
                84    0.0           1.0       0.0      0.0        0.0       0.0  0.0    0.0  1.308646
                90    0.0           1.0       0.0      0.0        1.0       0.0  0.0    0.0  1.007901
                108   0.0           1.0       1.0      0.0        0.0       0.0  0.0    0.0  1.308646
                114   0.0           1.0       1.0      0.0        1.0       0.0  0.0    0.0  1.007901
                name: (0.0, 0.0, 0.0, 0.0, 2.0)
                    size  neighborhood  children  schools  amenities  location  age  price       wts
                12   2.0           1.0       0.0      0.0        0.0       0.0  0.0    0.0  1.308646
                18   2.0           1.0       0.0      0.0        1.0       0.0  0.0    0.0  1.007901
                ......
            """

            wts_sum = new_df.groupby(var_pares)['wts'].agg(np.sum)
            """wts_sum:
                    price  location  age  schools  size     wts
                    0.0    0.0       0.0  0.0      0.0     4.633094
                                                   2.0     2.316547
                                          1.0      0.0     2.316547
                                     1.0  0.0      0.0     4.633094
                                                   2.0     2.316547
                                          1.0      0.0     2.316547
                           1.0       0.0  0.0      0.0     2.014696
                                                   2.0     1.007348
                                          1.0      0.0     1.007348
                                     1.0  0.0      0.0     2.014696
                                                   2.0     1.007348
                                          1.0      0.0     1.007348
            """
            # print('wts_sum:\n', wts_sum)
            bn_count = pd.DataFrame(wts_sum.index.tolist(), columns=var_pares)
            bn_count['count'] = wts_sum.tolist()

            bn_count['count'] = bn_count['count'].fillna(0.05)  # substitute 'NaN' of 'count' with 0.05

            # print('var_pares:' + str(var_pares) + '\nvar_pares counts:\n' + str(bn_count))
            # the columns of 'learn_cpds_df' include 'var pare p count', here we only need columns 'var pare p'
            bn['learn_cpds_df'][var] = pd.merge(bn['learn_cpds_df'][var].loc[:, var_pares + ['p']],
                                                bn_count, on=var_pares, how='outer')
            # replace 'NaN' of 'count' with 0.05
            bn['learn_cpds_df'][var]['count'] = bn['learn_cpds_df'][var]['count'].fillna(0.05)
            # print('\nlearn_cpds_df:' + str(var_pares) + '\n', bn['learn_cpds_df'][var])
            my_bayesnet.normalise_cpt(bn, var)
            get_learn_cpd(bn)


# DEBN
def Expectation_Maximisation(df, mis_index, bn, nodes, edges, var_card, size, Pre_PMI_IJK, K):
    """
        Dynamic embeddings for BN parameter learning (DEBN):
            1.E-step
            2.M-step
            3.New_PMI_IJK and Obj_SimChange
            4.calculate Loss_bound
              if (Loss_store(i + 1) >= (1 + Theta) * Loss_bound(i + 1))
                [U{i + 1}, S{i + 1}, V{i + 1}] = svds(New_PMI_IJK, K)
                U_cur = U{i+1} * sqrt(S{i+1})
                V_cur = V{i+1} * sqrt(S{i+1})
        """
    curr_iter = 1
    time_i = time.time()
    time_svd_all = []
    U, S, V = {}, {}, {}
    Loss_store = {}
    # N = Pre_PMI_IJK.shape[0]  # the number of rows
    PMI_IJK_index_name = Pre_PMI_IJK.index.tolist()
    N = len(PMI_IJK_index_name)

    # PMI_IJK_array = PMI_IJK.values
    u, s, v = svds(Pre_PMI_IJK.values, K)
    U[0], S[0], V[0] = np.abs(u[:, ::-1]), np.diag(np.abs(s[::-1])), np.abs(v[:, ::-1]).T
    U_cur = np.abs(np.dot(U[0], np.sqrt(S[0])))
    V_cur = np.abs(np.dot(V[0], np.sqrt(S[0])))
    Loss_store[0] = incremental_SVD.Obj(Pre_PMI_IJK.values, U_cur, V_cur)
    end_time_svd = time.time()
    time_svd_all.append(end_time_svd-time_i)

    while True:
        print("ITERATION #" + str(curr_iter))
        prev_cpts = []  # pre_cpts denotes the initialized cpt by init_param method
        prev_bn = bn
        # with open('./data/prev_bn_inter(' + str(curr_iter) + ').txt', 'w') as f:
        #     f.write(str(prev_bn))
        for X in bn['V']:
            prev_cpts.append(np.array(list(bn['learn_cpds_df'][X]['p'])))
        # print('prev_cpts:\n', prev_cpts)
        # print('location_cpt:\n', bn['learn_cpds_df']['location']['p'])
        # step0 = time.time()
        # print("STEP E: "+ str(curr_iter))
        new_df = Expectation(bn, df, mis_index, U_cur, V_cur, PMI_IJK_index_name)
        # print('new_df:\n', new_df)
        Maximisation(new_df, bn)

        new_bn = bn
        # with open('./data/new_bn_inter(' + str(curr_iter) + ').txt', 'w') as f:
        #     f.write(str(new_bn))

        new_cpts = []  # new_cpts denotes the learned cpt by EM algorithm
        for X in bn['V']:
            new_cpts.append(np.array(list(bn['learn_cpds_df'][X]['p'])))
        # print('new_location_cpt:\n', bn['learn_cpds_df']['location']['p'])
        # print('new_cpts:\n', new_cpts)
        diffs = []
        for i in range(len(prev_cpts)):
            max_diff = max(abs(np.subtract(prev_cpts[i], new_cpts[i])))  # subtract denotes the - operation
            diffs.append(max_diff)
        delta = max(diffs)
        time_f = time.time()
        print("Delta: " + str(delta))
        if (time_f - time_i) > 660:
            print("OVER TIME. . . . ")
            break
        if delta <= para_init.threshold_delta:
        # if delta <= 0.06:
            break

        start_time_update_svd = time.time()
        delta_PMI_IJK = my_bayesnet.delta_PMI_construction(bn, nodes, edges, var_card,
                                                           mis_index, size, PMI_IJK_index_name)
        New_PMI_IJK = Pre_PMI_IJK + delta_PMI_IJK
        # start_time_update_svd = time.time()
        U_temp_i = []
        U_temp_v = []
        for i in range(U[curr_iter - 1].shape[1]):
            temp_i_i = np.argmax(U[curr_iter - 1][:, i])
            U_temp_i.append(temp_i_i)
            temp_v_v = np.max(U[curr_iter - 1][:, i])
            U_temp_v.append(temp_v_v)
        V_temp_v = []

        V_j = 0
        for i in range(V[curr_iter - 1].shape[1]):
            if U_temp_i[V_j] or U_temp_i[V_j] == 0:
                V_temp_v.append(V[curr_iter - 1][U_temp_i[V_j], i])
                V_j = V_j + 1
                if V_j == len(U_temp_i):
                    break

        # print('V:\n', V[curr_iter - 1])
        # print('V_temp:\n', V_temp_v)
        temp_sign = np.sign(np.array(U_temp_v) * np.array(V_temp_v))
        Old_L = np.diag(S[curr_iter - 1]).T * temp_sign

        # calculate sum term
        temp_sum = np.dot(np.dot(U[curr_iter - 1].T, delta_PMI_IJK.values), U[curr_iter - 1])  # temp_sum = X^t
        Delta_L = np.diag(temp_sum).T
        New_S = np.diag(np.abs(Old_L + Delta_L))  # update eig <-- eig + delta_eig
        Delta_X = np.zeros((N, K))
        for i in range(K):
            # calculate v <-- lamda + delta_lamda - lamda, D <-- diag(v)
            temp_D = np.diag(np.dot(np.ones(K), (Old_L[i] + Delta_L[i])) - Old_L)
            try:
                temp_alpha = np.dot(np.linalg.inv(temp_D - temp_sum), temp_sum[:, i])
            except:
                temp_alpha = np.dot(temp_D - temp_sum, temp_sum[:, i])
            Delta_X[:, i] = np.dot(U[curr_iter - 1], temp_alpha)
        New_U = np.abs(U[curr_iter - 1] + Delta_X)  # new U

        New_V = np.dot(New_U, np.diag(np.sign(Old_L + Delta_L)))  # new V
        U[curr_iter], S[curr_iter], V[curr_iter] = New_U, New_S, New_V
        U_cur = np.dot(U[curr_iter], np.sqrt(S[curr_iter]))
        V_cur = np.dot(V[curr_iter], np.sqrt(S[curr_iter]))
        Loss_store[curr_iter] = incremental_SVD.Obj(New_PMI_IJK.values, U_cur, V_cur)

        end_time_update_svd = time.time()
        time_svd_all.append(end_time_update_svd - start_time_update_svd)
        curr_iter += 1
    print('Loss_store:', Loss_store)
    print('average_reconstruction_loss:', sum(Loss_store.values()) / curr_iter)
    print('average_time_svd:', np.mean(time_svd_all))
    print("Converged in (" + str(curr_iter) + ") iterations")
    return bn


if __name__ == '__main__':
    # # bn, df, mis_index = setup_network('data/network.bif', 'data/bn_data_10_multivar.xlsx')
    # bn, df, mis_index = setup_network('data/bn/alarm.bif', 'data/missingdata/500/alarm_500_TPR_CO.xlsx')
    # # print('bn:\n', bn)
    # # writer = BIFWriter(bn)
    # # writer.write_bif('data/network_initparam_written.bif')
    # edges = [tuple(edge) for edge in bn['E']]
    # nodes = bn['V']
    # """
    # edges = [['amenities', 'location'], ['neighborhood', 'location'],...]
    # """
    # var_card = bn['cardinality']
    # size = 1000
    # K = 4
    # Pre_PMI_IJK, edge_weight = my_bayesnet.PMI_construction(bn, nodes, edges, var_card, size)
    # network = Expectation_Maximisation(df, mis_index, bn, nodes, edges, var_card, size, Pre_PMI_IJK, K)

    # bn_name = ['child', 'alarm', 'hepar2', 'andes', 'munin1', 'link']
    # bn_data = ['data/bn/child.bif', 'data/bn/alarm.bif', 'data/bn/hepar2.bif',
    #            'data/bn/andes.bif', 'data/bn/munin1.bif', 'data/bn/link.bif']

    bn_name = para_init.bn_name
    bn_data = para_init.bn_data
    missingdata = para_init.missingdata

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

    # # two latent variables
    # missingdata = ['data/missingdata/500/child_500_disease_LungParench.xlsx', 'data/missingdata/500/alarm_500_TPR_CO.xlsx',
    #                'data/missingdata/500/hepar2_500_surgery_fat.xlsx', 'data/missingdata/500/andes_500_SNode_3_SNode_4.xlsx',
    #                'data/missingdata/500/munin1_500_MUDENS_MALOSS.xlsx', 'data/missingdata/500/link_500_N73.xlsx']

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

    size = 1000
    K = 4
    # K = 24
    # K = 32

    for i in range(len(bn_name)):
        print(30 * "***")
        print('bn_name:', bn_name[i])
        start_time = time.time()
        bn, df, mis_index = setup_network(bn_data[i], missingdata[i])

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


        # print('bn:\n', bn)
        # writer = BIFWriter(bn)
        # writer.write_bif('data/network_initparam_written.bif')
        edges = [tuple(edge) for edge in bn['E']]
        nodes = bn['V']
        """
        edges = [['amenities', 'location'], ['neighborhood', 'location'],...]
        """
        var_card = bn['cardinality']
        Pre_PMI_IJK, edge_weight = my_bayesnet.PMI_construction(bn, nodes, edges, var_card, size)
        network = Expectation_Maximisation(df, mis_index, bn, nodes, edges, var_card, size, Pre_PMI_IJK, K)
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












