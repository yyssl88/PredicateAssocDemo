import numpy as np
import pandas as pd
import copy
import math
import time
import sys

class PAssoc(object):
    def __init__(self, supp_taus, data_file):
        super(PAssoc, self).__init__()
        self.source_data = None
        self.attrs = []
        self.supp_taus = supp_taus
        self.current_state = None
        self.satisfied_tuples = {}
        self.loadData(data_file)
        self.buildPLI()

    def loadData(self, data_file):
        self.source_data = pd.read_csv(data_file)
        # use attr to represent predicates
        self.attrs = list(self.source_data.columns.values)
        self.current_state = np.zeros((len(self.attrs)))

    def getPredicateNum(self):
        return len(self.attrs)

    def reset(self):
        # return observation with zero predicate
        self.current_state = np.zeros(len(self.attrs))
        return copy.deepcopy(self.current_state)

    def buildPLI(self):
        for attr in self.attrs:
            self.satisfied_tuples[attr] = []
            for value, df in self.source_data.groupby(attr):
                if df.shape[0] == 1:  # remove this for partial order predicates.
                    continue
                self.satisfied_tuples[attr].append(df.index.to_list())

    def cal_supp(self, attr_set, satisfied_tuples):
        # save the id of tuples that satisfy the same attribute of attr_set
        res = satisfied_tuples[attr_set[0]]
        for i in range(1, len(attr_set)):
            new_res = []
            for tid_set in res:
                for tid_set2 in satisfied_tuples[attr_set[i]]:
                    intersection = list(set(tid_set).intersection(set(tid_set2)))
                    if len(intersection) <= 1:  # change this for partial order predicates.
                        continue
                    new_res.append(intersection)
            res = new_res
        # calculate support
        if len(res) == 0:
            return 0
        supp = 0
        m = 2
        for tid_set in res:
            n = len(tid_set)
            supp += math.factorial(n) // (math.factorial(m) * math.factorial(n - m))  # C(n, m)
        return supp

    def transformAttr(self, state):
        attr_arr = []
        for i, e in enumerate(state):
            if e == 1:
                attr_arr.append(self.attrs[i])
        return attr_arr

    def step(self, action):
        base_action = np.zeros((len(self.attrs)))
        base_action[action] = 1.0
        # next state
        next_state = copy.deepcopy(self.current_state)
        next_state[action] = 1.0

        # reward function
        attr_arr = self.transformAttr(next_state)
        if len(attr_arr) == 0:
            supp = 0;
        else:
            supp = self.cal_supp(attr_arr, self.satisfied_tuples)
        reward = supp - self.supp_taus

        done = False
        if reward <= 0:
            done = True

        # update state
        self.current_state = next_state
        return copy.deepcopy(self.current_state), reward, done


