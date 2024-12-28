import networkx as nx
from copy import deepcopy
from itertools import product
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import pandas as pd
import networkx as nx
from decimal import *

import re
import collections
from string import Template
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore, \
    cppStyleComment, printables


class BIFReader:
    """
    Base class for reading network file in bif format
    """

    def __init__(self, path=None, string=None):
        """
        # dog-problem.bif file is present at
        # http://www.cs.cmu.edu/~javabayes/Examples/DogProblem/dog-problem.bif
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader = BIFReader("bif.bif")
        """

        if path:
            with open(path, "r") as network:
                self.network = network.read()
                # print(type(self.network)) # str
                # print(self.network)
                # exit(0)

        elif string:
            self.network = string  # such as variable amenities {type discrete [ 2 ] { lots, little };}

        if '"' in self.network:
            # Replacing quotes by spaces to remove case sensitivity like:
            # "Dog-Problem" and Dog-problem
            # or "true""false" and "true" "false" and true false
            self.network = self.network.replace('"', ' ')

        if '/*' in self.network or '//' in self.network:
            self.network = cppStyleComment.suppress().transformString(self.network)  # removing comments from the file
        # print(type(self.network))
        # print('network:\n', self.network)

        self.name_expr, self.state_expr, self.property_expr = self.get_variable_grammar()
        self.probability_expr, self.cpd_expr, self.cpd_state_expr = self.get_probability_grammar()
        self.network_name = self.get_network_name()
        self.variable_names = self.get_variables()
        # print(self.variable_names)
        self.variable_states = self.get_states()
        # print(self.variable_states)
        self.variable_card = self.get_cardinality()
        self.variable_parents, self.variable_children = self.get_parents_and_children()
        self.variable_mb = self.get_markov_blanket()
        self.variable_cpds = self.get_my_cpt()
        # print(self.variable_cpds)
        self.variable_cpd_df = self.get_cpt_df()
        # for v in self.variable_cpd_df:
        #     print(self.variable_cpd_df[v])
        self.variable_edges = self.get_edges()
        self.topological_variable = self.topological_nodes()
        # print(self.topological_variable)
        # exit(0)

    def get_variable_grammar(self):
        """
         A method that returns variable grammar
         variable amenities {
            type discrete [ 2 ] { lots, little };
        }
        """
        # Defining a expression for valid word
        word_expr = Word(alphanums + '_' + '-')
        name_expr = Suppress('variable') + word_expr + Suppress('{')  # Suppress 忽略表达式里的内容
        word_expr2 = Word(initChars=printables, excludeChars=['{', '}', ',', ' '])
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        # Defining a variable state expression
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) + \
                              Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
        # 使用group将返回的结果，使匹配的合成一个字符串
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = Suppress('property') + CharsNotIn(';') + Suppress(';')  # Creating a expr to find property

        return name_expr, variable_state_expr, property_expr

    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        probability ( location | amenities, neighborhood ) {
                          (lots, bad) 0.3, 0.4, 0.3;
                          (lots, good) 0.8, 0.15, 0.05;
                          (little, bad) 0.2, 0.4, 0.4;
                          (little, good) 0.5, 0.35, 0.15;
        }
        """
        # Creating valid word expression for probability, it is of the format
        # wor1 | var2 , var3 or var1 var2 var3 or simply var
        word_expr = Word(alphanums + '-' + '_') + Suppress(Optional("|")) + Suppress(Optional(","))
        word_expr2 = Word(initChars=printables, excludeChars=[',', ')', ' ', '(']) + Suppress(Optional(","))
        # creating an expression for valid numbers, of the format
        # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
        num_expr = Word(nums + '-' + '+' + 'e' + 'E' + '.') + Suppress(Optional(","))
        probability_expr = Suppress('probability') + Suppress('(') + OneOrMore(word_expr) + Suppress(')')
        optional_expr = Suppress('(') + OneOrMore(word_expr2) + Suppress(')')
        probab_attributes = optional_expr | Suppress('table')
        cpd_expr = Suppress(probab_attributes) + OneOrMore(num_expr)
        cpd_state_expr = probab_attributes + OneOrMore(num_expr)
        # print('cpd_expr:\n', cpd_expr)
        return probability_expr, cpd_expr, cpd_state_expr

    def variable_block(self):
        start = re.finditer('variable', self.network)
        for index in start:  # iterator <re.Match object; span=(442, 450), match='variable'>
            end = self.network.find('}\n', index.start())  # index.start()=442, end = 140
            yield self.network[index.start():end]  # yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器

    def probability_block(self):
        start = re.finditer('probability', self.network)
        for index in start:
            end = self.network.find('}\n', index.start())
            yield self.network[index.start():end]
            # yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器
            # 生成器同样是可迭代对象，但是你只能读取一次，因为它并没有把所有值存放内存中，它动态的生成值

    def get_network_name(self):
        """
        Retruns the name of the network

        Example
        ---------------
        from pgmpy.readwrite import BIFReader
        reader = BIF.BifReader("bif_test.bif")
        reader.network_name()
        'Dog-Problem'
        """
        start = self.network.find("network")
        end = self.network.find("}\n", start)
        # Creating a network attribute
        network_attribute = Suppress("network") + Word(alphanums + "_" + "-") + "{"
        network_name = network_attribute.searchString(self.network[start:end])[0][0]

        return network_name

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_variables()
        ['light-on','bowel_problem','dog-out','hear-bark','family-out']
        """
        variable_names = []
        # print('variable_block:\n', self.variable_block())
        for block in self.variable_block():  # self.variable_block() is a generator, which is generated by yield
            # print('block:', block)
            name = self.name_expr.searchString(block)[0][0]
            variable_names.append(name)
        # print(variable_names)
        # exit(0)
        return variable_names

    def get_states(self):
        """
        Returns the states of variables present in the network

        Example
        -----------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_states()
        {'bowel-problem': ['true','false'],
        'dog-out': ['true','false'],
        'family-out': ['true','false'],
        'hear-bark': ['true','false'],
        'light-on': ['true','false']}
        """
        variable_states = {}
        # variable_cardinality = {}
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            variable_states[name] = list(self.state_expr.searchString(block)[0][0])
            # variable_cardinality[name] = len(variable_states[name])
            # print('search_state_expr:', self.state_expr.searchString(block))

        return variable_states

    def get_cardinality(self):
        variable_card = {variable: len(states) for variable, states in self.variable_states.items()}
        return variable_card

    def get_parents_and_children(self):
        """
        Returns the parents and children of the variables present in the network

        Example
        --------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_parents()
        {'bowel-problem': [],
        'dog-out': ['family-out', 'bowel-problem'],
        'family-out': [],
        'hear-bark': ['dog-out'],
        'light-on': ['family-out']}
        """
        variable_parents = {}
        variable_children = {self.variable_names[i]: [] for i in range(len(self.variable_names))}
        for block in self.probability_block():
            names = self.probability_expr.searchString(block.split('\n')[0])[0]
            variable_parents[names[0]] = names[1:]
            if len(names) > 1:  # only one variable, meaning that no parents
                for i in range(1, len(names)):  # starting from name[1], name[1] represents the first parent
                    if names[0] not in variable_children[names[i]]:
                        # represent the node name[0] not existing its parents
                        variable_children[names[i]].append(names[0])
        # print('variable_parents:', variable_parents)
        # print('variable_children:', variable_children)
        return variable_parents, variable_children

    def get_markov_blanket(self):
        """
        variable_mb: {'amenities': ['location', 'neighborhood'],
                    'neighborhood': ['location', 'children', 'amenities'],
                    'location': ['amenities', 'neighborhood', 'age', 'price', 'schools', 'size'],
                    'children': ['neighborhood', 'schools'], ...}
        return Markov blanket
        """
        variable_mb = {self.variable_names[i]: [] for i in range(len(self.variable_names))}  # initialize mb(var)
        for var, value in variable_mb.items():  # traverse the variable
            var_pare = self.variable_parents[var]  # get the parents of variable
            var_children = self.variable_children[var]  # get the children of variable
            value += var_pare + var_children  # add the parents and children of variable to the MB
            for var_child in var_children:
                var_spouses = self.variable_parents[var_child]  # get the spouses of variable
                for var_spouse in var_spouses:
                    if var_spouse not in value and var_spouse != var:  # if spouse not in MB(var) and ！= var
                        value.append(var_spouse)  # add the spouse to the MB(var)
        # print('variable_mb:', variable_mb)
        return variable_mb

    def get_values(self):
        """
        Returns the CPD of the variables present in the network
        probability ( location | amenities, neighborhood ) { // location ( good, bad, ugly)
                          (lots, bad) 0.3, 0.4, 0.3;         // amenities (lots, little)
                          (lots, good) 0.8, 0.15, 0.05;     // neighborhood(bad, good)
                          (little, bad) 0.2, 0.4, 0.4;
                          (little, good) 0.5, 0.35, 0.15;}
        Example
        --------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_values()
        {'amenities': array([[0.3],[0.7]]),
       'location': array([[0.3 , 0.8 , 0.2 , 0.5 ],
                          [0.4 , 0.15, 0.4 , 0.35],
                          [0.3 , 0.05, 0.4 , 0.15]]),....}
         """
        variable_cpds = {}
        for block in self.probability_block():
            name = self.probability_expr.searchString(block)[0][0]  # names:amenities ...
            # print('name:\n', name)
            cpds = self.cpd_expr.searchString(block)  # cpd_expr does not include the state combination
            # print('cpds:\n', cpds)
            """
            children_cpts: [['0.6', '0.4'], ['0.3', '0.7']]
            """
            # arr = [round(float(j), 3) for i in cpds for j in i]
            arr = [j for i in cpds for j in i]
            # print('arr:\n', arr)
            if 'table' in block:
                arr = np.array(arr)
                arr = arr.reshape((len(self.variable_states[name]),
                                   arr.size // len(self.variable_states[name])))
            else:
                length = len(self.variable_states[name])
                reshape_arr = [[] for i in range(length)]
                for i, val in enumerate(arr):
                    reshape_arr[i % length].append(val)
                arr = reshape_arr
                arr = np.array(arr)
            variable_cpds[name] = arr
        # print('variable_cpds:\n', variable_cpds)
        return variable_cpds

    def get_my_cpt(self):
        """
        Returns the CPD of the variables present in the network
        probability ( location | amenities, neighborhood ) { // location ( good, bad, ugly)
                          (lots, bad) 0.3, 0.4, 0.3;         // amenities (lots, little)
                          (lots, good) 0.8, 0.15, 0.05;     // neighborhood(bad, good)
                          (little, bad) 0.2, 0.4, 0.4;
                          (little, good) 0.5, 0.35, 0.15;}
        Example
        --------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_my_cpt()
        {'amenities': array([[0.3],[0.7]]),
       'location': array([[0.3 ,  0.4 , 0.3 ],
                          [0.8 , 0.15, 0.05 ],
                          [0.2 ,  0.4, 0.4  ],
                          [0.5 , 0.35, 0.15 ]),....}
         """
        variable_cpds = {}
        # print(self.probability_block())
        for block in self.probability_block():
            # print(block)

            name = self.probability_expr.searchString(block)[0][0]  # names:amenities ...
            # print('name:\n', name)
            cpds = self.cpd_expr.searchString(block)  # cpd_expr does not include the state combination
            # print('cpds:\n', cpds)
            """
            children_cpts: [['0.6', '0.4'], ['0.3', '0.7']]
            """
            import para_init
            # arr = [round(float(j), 3) for i in cpds for j in i]
            # arr = [np.round(float(j), 1) for i in cpds for j in i]
            # arr = [np.round(float(j), para_init.read_cpd_rould_decimal) for i in cpds for j in i]
            # print(para_init.read_cpd_rould_decimal)
            # print(para_init.read_cpd_rould)
            if para_init.read_cpd_rould_decimal == None and para_init.read_cpd_rould == None:
                arr = [Decimal(j) for i in cpds for j in i]
                # print(arr)
                # exit(0)
            elif para_init.read_cpd_rould_decimal != None:
                arr = [round(Decimal(j), para_init.read_cpd_rould_decimal) for i in cpds for j in i]
            elif para_init.read_cpd_rould != None:
                arr = [np.round(float(j), para_init.read_cpd_rould) for i in cpds for j in i]
            # print('arr111:\n', arr)
            # print(sum(arr))

            arr = np.array(arr)
            # print('arr:\n', arr)

            if 'table' not in block:
                arr = arr.reshape((arr.size // len(self.variable_states[name]),
                                   len(self.variable_states[name])))
            # print('arr:\n', arr)
            #  make the sum of arr == 1
            # if isinstance(arr[0], np.ndarray):  # if arr[0] is ndarray, we add the all elements of array
            #     for i in range(len(arr)):
            #         i_sum = np.sum(arr[i])
            #         if i_sum != 1:
            #             # i_temp = []
            #             # print('name:', name)
            #             # print('i:', arr[i])
            #             i_temp = [j/i_sum for j in arr[i]]
            #             # print('i_temp:', i_temp)
            #             arr[i] = i_temp
            # else:
            #     arr_sum = np.sum(arr)
            #
            #     arr = [j/arr_sum for j in arr]
            if isinstance(arr[0], np.ndarray):  # if arr[0] is ndarray, we add the all elements of array
                for i in range(len(arr)):
                    i_sum = np.sum(arr[i])
                    if i_sum == 0:
                        i_temp = [1.0 / len(arr[i]) for j in arr[i]]
                        arr[i] = i_temp
                    elif i_sum != 1:
                        # i_temp = []
                        # print('name:', name)
                        # print('i:', arr[i])
                        i_temp = [j/i_sum for j in arr[i]]
                        # print('i_temp:', i_temp)
                        arr[i] = i_temp
                        # exit(0)

            else:
                arr_sum = np.sum(arr)
                if arr_sum == 0:
                    arr = [1.0 / len(arr) for j in arr]
                else:
                    arr = [j / arr_sum for j in arr]
            # print('arr:\n', arr)


            variable_cpds[name] = arr
            # print('variable_cpds:\n', variable_cpds)

            # if name == "HypDistrib":
            #     exit(0)

        # exit(0)
        return variable_cpds

    # get each variable's cpt like dataframe
    def get_cpt_df(self):
        """
        {'amenities':    amenities    p
                    0          0  0.3
                    1          1  0.7,
                    'neighborhood':
                        neighborhood    p
                    0          0      0.4
                    1          1      0.6,
                    'location':
                        location  amenities  neighborhood     p
                    0          0          0             0  0.30
                    1          1          0             0  0.40
                    2          2          0             0  0.30
                    3          0          0             1  0.80
                    4          1          0             1  0.15
                    5          2          0             1  0.05
                    6          0          1             0  0.20
                    7          1          1             0  0.40
                    8          2          1             0  0.40
                    9          0          1             1  0.50
                    10         1          1             1  0.35
                    11         2          1             1  0.15,}
        """
        cpt_df = {var: pd.DataFrame() for var in self.variable_names}
        # print(cpt_df)
        for var in self.variable_names:
            pares = self.variable_parents[var]
            # print(pares)
            if not pares:  # variable does not have parents
                var_card = self.variable_card[var]
                # print(var_card)
                probability = self.variable_cpds[var]
                # print(probability)
                cpt_df[var] = pd.DataFrame({var: range(0, var_card), 'p': probability})
                # print(cpt_df[var])
                # exit(0)
            else:
                var_pares = [var] + pares  # variable has parents, var_pares includes var and parents
                # print('var_pares:', var_pares) # var_pares: ['HypDistrib', 'DuctFlow', 'CardiacMixing']
                var_card = [self.variable_card[var] for var in var_pares]  # get each cardinality of variable
                # print('var_card:', var_card) # var_card: [2, 3, 4]

                # print(self.variable_cpds)
                # exit(0)

                probability = self.variable_cpds[var].ravel()  # convert multi-array to one array
                # print(probability) # [1.  0.  1.  0.  0.  1.  1.  0.  1.  0.  0.5 0.5 1.  0.  1.  0.  1.  0. 1.  0.  1.  0.  0.5 0.5]
                state_combs = np.array(list(product(*[range(i) for i in var_card])))  # get state combinations
                # print([range(i) for i in var_card]) # [range(0, 2), range(0, 3), range(0, 4)]
                # print(*[range(i) for i in var_card]) # range(0, 2) range(0, 3) range(0, 4)
                # print(list(product(*[range(i) for i in var_card])))
                # [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0),
                #  (0, 2, 1), (0, 2, 2), (0, 2, 3), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0), (1, 1, 1),
                #  (1, 1, 2), (1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3)]
                # print(state_combs)
                # [[0 0 0]
                #  [0 0 1]
                #  [0 0 2]
                #  [0 0 3]
                #     ...
                #  [1 2 3]]
                # exit(0)
                # transform state combination into dataframe, columns is named as var_name, sorted as parents
                ####################################################################################################
                # cpt_df[var] = pd.DataFrame(state_combs, columns=var_pares).sort_values(by=pares, ascending=True)
                ####################################################################################################
                def Reverse(lst):
                    return [ele for ele in reversed(lst)]
                cpt_df[var] = pd.DataFrame(state_combs, columns=var_pares).sort_values(by=Reverse(pares), ascending=True)
                cpt_df[var][var] = [a for a in range(var_card[0])] * int((len(cpt_df[var][var]) / var_card[0]))
                ####################################################################################################
                # HypDistrib DuctFlow CardiacMixing
                # 0  0 0 0
                # 12 0 1 2
                # 1  1 0 0
                # ...
                # 23 1 2 3
                cpt_df[var] = cpt_df[var].reset_index(drop=True)  # reorder the index of the rows
                # print(cpt_df[var])
                # HypDistrib DuctFlow CardiacMixing
                # 1  0 0 0
                # 2  0 1 2
                # 3  1 0 0
                # ...
                # 23 1 2 3
                cpt_df[var]['p'] = probability  # append the columns 'p' into the cpt_df
                # print(cpt_df[var])
                # HypDistrib DuctFlow CardiacMixing p
                # 1  0 0 0 1.0
                # 2  0 1 2 0.0
                # 3  1 0 0 1.0
                # ...
                # 23 1 2 3 0.5
                # exit(0)
        # print(cpt_df['R_LNLW_MED_SEV'])
        # exit(0)
        return cpt_df



    def get_edges(self):
        """
        Returns the edges of the network

        Example
        --------
        #>>> from pgmpy.readwrite import BIFReader
        #>>> reader = BIFReader("bif.bif")
        #>>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edges = [
            [parent, child]
            for child in self.variable_parents.keys()
            for parent in self.variable_parents[child]
        ]
        return edges

    def topological_nodes(self):  # eg denotes the edges of G
        G = nx.DiGraph()
        G.add_edges_from(self.variable_edges)
        # G_edges = [e for e in G.edges_iter()]
        # print('G_edges:\n', G_edges)
        topological_node = list(nx.topological_sort(G))
        return topological_node

    def my_model(self):
        # print('self_variable_parents', self.variable_parents)
        net = {}
        net.setdefault('VN', self.variable_names)
        net.setdefault('V', self.topological_variable)
        net.setdefault('E', self.variable_edges)
        net.setdefault('parents', self.variable_parents)
        net.setdefault('children', self.variable_children)
        net.setdefault('mb', self.variable_mb)
        net.setdefault('states', self.variable_states)
        net.setdefault('cardinality', self.variable_card)
        net.setdefault('cpds', self.variable_cpds)
        net.setdefault('cpds_df', self.variable_cpd_df)
        net.setdefault('learn_cpds_df', deepcopy(self.variable_cpd_df))
        net.setdefault('learn_cpds', None)

        return net


class BIFWriter:
    """
    Base class for writing BIF network file format
    """

    def __init__(self, bn):
        """
        Initialise a BIFWriter Object

        Parameters
        ----------
        model: BayesianModel Instance

        Examples
        ---------
        writer = BIFWriter(BN)
        writer.write_bif('data/network_written.bif')
        """

        self.bn = bn
        self.network_name = "network"
        self.variable_states = self.bn['states']
        self.variable_parents = self.bn['parents']
        self.tables = self.get_cpd_value()

    def BIF_templates(self):
        """
        Create template for writing in BIF format
        """
        network_template = Template("network $name {\n}\n")
        # property tag may or may not be present in model,and since no of properties
        # can be more than one , will replace them accoriding to format otherwise null
        variable_template = Template(
            """variable $name {\n type discrete [ $no_of_states ] { $states };\n}\n"""
        )
        # $variable_ here is name of variable, used underscore for clarity
        probability_template = Template(
            """probability ( $variable_ ) {\n table $values ;\n}\n"""
        )
        probability_template_pares = Template(
            """probability ( $variable_$seprator_$parents ) {\n$values}\n"""
        )  # Template("""probability ( $variable_ ) {\n table $values ;\n}\n""")
        values_template = Template(""" ($state) $value;\n""")
        return (
            network_template,
            variable_template,
            probability_template,
            probability_template_pares,
            values_template
        )

    def __str__(self):
        """
        Returns the BIF format as string
        """
        network_template, variable_template, probability_template, \
        probability_template_pares, values_template = (self.BIF_templates())
        network = ""
        network += network_template.substitute(name=self.network_name)
        variables = self.bn['V']

        for var in variables:
            no_of_states = str(len(self.variable_states[var]))
            states = ", ".join(self.variable_states[var])
            network += variable_template.substitute(
                name=var,
                no_of_states=no_of_states,
                states=states,
            )

        for var in variables:
            if not self.variable_parents[var]:
                temp_cpd = self.tables[var]
                # temp_cpd[1] = 0.8
                # print(temp_cpd)
                if sum(temp_cpd) == 1.0:
                    pass
                else:
                    index_max = np.where(temp_cpd == np.max(temp_cpd))
                    # print(index_max)
                    temp_cpd[index_max] = 0
                    temp_cpd[index_max] = 1.0 - sum(temp_cpd)
                    # print(temp_cpd)
                # print(sum(temp_cpd))
                # exit(0)
                cpd = ", ".join(map(str, temp_cpd))
                # cpd = ", ".join(map(str, self.tables[var]))

                network += probability_template.substitute(
                    variable_=var, values=cpd
                )

            else:
                #################################################################################
                list_variable_parents = [p for p in self.variable_parents[var]]
                # print(list_variable_parents)
                # exit(0)
                pares = ", ".join(list_variable_parents)
                # print(pares)
                #################################################################################
                # pares = ", ".join(self.variable_parents[var])
                #################################################################################
                seprator = " | "
                cpd_values = ""
                #################################################################################
                def Reverse(lst):
                    return [ele for ele in reversed(lst)]
                # print(list_variable_parents)
                Reverse_list_variable_parents = Reverse(list_variable_parents)
                # Reverse_list_variable_parents = list_variable_parents
                # print(Reverse_list_variable_parents)
                # print(self.variable_states)
                state_combs = np.array(list(product(*[self.variable_states[pares]
                                                      for pares in Reverse_list_variable_parents])))
                # print(state_combs)
                ############################################################################
                # state_combs = np.array(list(product(*[self.variable_states[pare]
                #                                       for pare in self.variable_parents[var]])))
                # print(self.tables[var])
                ############################################################################
                for states_comb, valu in zip(state_combs, self.tables[var]):
                    temp_cpd = valu
                    if sum(temp_cpd) == 1.0:
                        pass
                    else:
                        # print(temp_cpd)
                        # print(sum(temp_cpd))
                        index_max = np.where(temp_cpd == np.max(temp_cpd))
                        # print(index_max)
                        temp_cpd[index_max] = 0
                        temp_cpd[index_max] = 1.0 - sum(temp_cpd)
                        # print(temp_cpd)
                        # print(sum(temp_cpd))
                        # exit(0)

                    # print(states_comb)
                    # print(valu)
                    states_combines = ", ".join(map(str, Reverse(states_comb)))
                    cpd_value = ", ".join(map(str, temp_cpd))
                    # cpd_value = ", ".join(map(str, valu))
                    cpd_values += values_template.substitute(state=states_combines, value=cpd_value)
                ############################################################################
                # for states_comb, valu in zip(state_combs, self.tables[var]):
                #     states_combines = ", ".join(map(str, states_comb))
                #     cpd_value = ", ".join(map(str, valu))
                #     cpd_values += values_template.substitute(state=states_combines, value=cpd_value)
                ############################################################################
                network += probability_template_pares.substitute(
                    variable_=var, seprator_=seprator, parents=pares, values=cpd_values
                )

        return network

    def str_rename(self, dict_V, dict_states):
        """
        Returns the BIF format as string
        """
        network_template, variable_template, probability_template, \
        probability_template_pares, values_template = (self.BIF_templates())
        network = ""
        network += network_template.substitute(name=self.network_name)
        variables = self.bn['V']

        # print(dict_V)
        # print(dict_states)
        # print(self.variable_states)
        # print(dict_states['1'])
        # exit(0)

        for var in variables:
            index = dict_V[var]
            # print(index)
            # exit(0)
            no_of_states = str(len(dict_states[index]))
            # print(no_of_states)
            states = ", ".join(dict_states[index])
            # print(states)
            network += variable_template.substitute(
                name=index,
                no_of_states=no_of_states,
                states=states,
            )
        # print(network)
        # exit(0)

        for var in variables:
            index = dict_V[var]
            # print(self.variable_parents[var])
            # exit(0)
            if not self.variable_parents[var]:
                cpd = ", ".join(map(str, self.tables[var]))
                network += probability_template.substitute(
                    variable_=index, values=cpd
                )
                # print(network)
                # exit(0)

            else:
                # print(123)
                # print(self.variable_parents[var])
                list_variable_parents = [dict_V[p] for p in self.variable_parents[var]]
                # print(list_variable_parents)
                # exit(0)
                pares = ", ".join(list_variable_parents)
                # print(pares)
                seprator = " | "
                cpd_values = ""
                ##############################################################################################
                # print(self.variable_states)
                # state_combs = np.array(list(product(*[self.variable_states[pare]
                #                                       for pare in self.variable_parents[var]])))
                #################################################################################
                def Reverse(lst):
                    return [ele for ele in reversed(lst)]
                # print(list_variable_parents)
                Reverse_list_variable_parents = Reverse(list_variable_parents)
                # print(Reverse_list_variable_parents)
                # exit(0)
                #################################################################################
                state_combs = np.array(list(product(*[dict_states[pares]
                                                      for pares in Reverse_list_variable_parents])))

                ############################################################################
                # cpt_df[var] = pd.DataFrame(state_combs, columns=var_pares).sort_values(by=Reverse(pares),
                #                                                                        ascending=True)
                # print(state_combs)
                # print(self.tables[var])
                # exit(0)
                # print(self.tables[var])
                for states_comb, valu in zip(state_combs, self.tables[var]):
                    # print(states_comb)
                    # print(valu)
                    # exit(0)
                    states_combines = ", ".join(map(str, Reverse(states_comb)))
                    cpd_value = ", ".join(map(str, valu))
                    cpd_values += values_template.substitute(state=states_combines, value=cpd_value)
                # print(cpd_values)
                # exit(0)

                network += probability_template_pares.substitute(
                    variable_=index, seprator_=seprator, parents=pares, values=cpd_values
                )
                # print(network)
                # exit(0)

                # if var == "HypoxiaInO2":
                #     for states_comb, valu in zip(state_combs, self.tables[var]):
                #         print(states_comb)
                #         states_combines = ", ".join(map(str, Reverse(states_comb)))
                #
                #         print(states_combines)
                #         print(valu)
                #         print(sum(valu))
                #         cpd_value = ", ".join(map(str, valu))
                #         print(cpd_value)
                #         for c in valu:
                #             print(c)
                #             print(str(c))
                #     print(cpd_values)
                #     # print(network)
                #     exit(0)

        # print(network)
        # exit(0)

        return network


    def get_cpd_value(self):
        """
        Adds tables to BIF

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        bn, df, mis_index = setup_network('data/network.bif', 'data/bn_data_1000.xlsx')
        bn/network = Expectation_Maximisation(df, bn, mis_index)
        writer = BIFWriter(bn/network)
        writer.get_cpd_value()
        {'bowel-problem': array([ 0.01,  0.99]),
         'dog-out': array([ 0.99,  0.97,  0.9 ,  0.3 ,  0.01,  0.03,  0.1 ,  0.7 ]),
         'family-out': array([ 0.15,  0.85]),
         'hear-bark': array([ 0.7 ,  0.01,  0.3 ,  0.99]),
         'light-on': array([ 0.6 ,  0.05,  0.4 ,  0.95])}
        """
        cpds = self.bn['learn_cpds_df']
        # for cpd in cpds:
        #     print(cpd)
        #     print(cpds[cpd])
        #     print(sum(cpds[cpd]['p']))
        # exit(0)
        variables = self.bn['V']
        tables = {}  # tables: {'amenities': array([0.302, 0.698]), 'neighborhood': array([0.426, 0.574])}
        for var in variables:
            # print(cpds[var])
            cpd_value = cpds[var]['p']  # get the probability of var
            # print(cpd_value)
            cpd_value = np.array(cpd_value)
            # print(cpd_value)
            pares = self.bn['parents'][var]
            # print(pares)
            # exit(0)
            if pares:
                cpd_value = cpd_value.reshape(cpd_value.size // len(self.variable_states[var]),
                                              len(self.variable_states[var]))  # reshape the np.array
            tables[var] = cpd_value
        #     print('tables:', tables)
        # exit(0)
        return tables

    def write_bif(self, filename):
        """
        Writes the BIF data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        from pgmpy.readwrite import BIFReader, BIFWriter
        model = BIFReader('dog-problem.bif').get_model()
        writer = BIFWriter(model)
        writer.write_bif('data/network_written.bif')
        """
        writer = self.__str__()
        with open(filename, "w") as fout:
            fout.write(writer)

    def write_bif_rename(self, filename, list_V, dict_states):
        writer = self.str_rename(list_V, dict_states)
        with open(filename, "w") as fout:
            fout.write(writer)


# reader = BIFReader('data/network.bif')  #network
# reader = BIFReader('data/dataset/alarm.bif')  #Alarm
# reader = BIFReader('data/dataset/hepar2.bif')   #Large BN

if __name__ == '__main__':
    reader = BIFReader('../data/network.bif')  # Very large BN
    net = reader.my_model()
    # print('net:\n', net)
