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
        #return copy.deepcopy(self.current_state)

    def initialAction(self, random_seed):
        action = np.random.randint(0, len(self.attrs))
        self.current_state[action] = 1.0
        return copy.deepcopy(self.current_state)

    def buildPLI(self):
        for attr in self.attrs:
            self.satisfied_tuples[attr] = []
            for value, df in self.source_data.groupby(attr):
                if df.shape[0] == 1:  # remove this for partial order predicates.
                    continue
                self.satisfied_tuples[attr].append(df.index.to_list())

    # attr_set: X; attr_rhs: Y
    def cal_supp(self, attr_set, attr_rhs, satisfied_tuples):
        # save the id of tuples that satisfy the same attribute of attr_set
        res_lhs = satisfied_tuples[attr_set[0]]
        for i in range(1, len(attr_set)):
            new_res = []
            for tid_set in res_lhs:
                for tid_set2 in satisfied_tuples[attr_set[i]]:
                    intersection = list(set(tid_set).intersection(set(tid_set2)))
                    if len(intersection) <= 1:  # change this for partial order predicates.
                        continue
                    new_res.append(intersection)
            res_lhs = new_res

        # save the id of tuples that satisfy the same attribute of both attr_set and attr_rhs
        res_rule = []
        if attr_rhs != "":
            for tid_set in res_lhs:
                for tid_set2 in satisfied_tuples[attr_rhs]:
                    intersection = list(set(tid_set).intersection(set(tid_set2)))
                    if len(intersection) > 1:  # change this for partial order predicates.
                        res_rule.append(intersection)
        # print("res_lhs:", res_lhs)
        # print("res_rule:", res_rule)

        # calculate support
        if len(res_lhs) == 0:
            return 0, 0, 0
        m = 2
        lhs_supp = 0
        for tid_set in res_lhs:
            n = len(tid_set)
            lhs_supp += math.factorial(n) // (math.factorial(m) * math.factorial(n - m))  # C(n, m)
        rule_supp = 0
        for tid_set in res_rule:
            n = len(tid_set)
            rule_supp += math.factorial(n) // (math.factorial(m) * math.factorial(n - m))  # C(n, m)
        conf = rule_supp * 1.0 / lhs_supp
        print(attr_set, "->", attr_rhs, ", lhs_supp:", lhs_supp * 2, ", supp:", rule_supp * 2, ", conf:", conf, "\n")
        return lhs_supp, rule_supp, conf


    def transformAttr(self, state):
        attr_arr = []
        for i, e in enumerate(state):
            if e == 1:
                attr_arr.append(self.attrs[i])
        return attr_arr

    def transformAttrInverse(self, attr_arr):
        state = np.zeros(len(self.attrs))
        for e in attr_arr:
            state[e] = 1.0
        return state

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
            supp, null, null = self.cal_supp(attr_arr, "", self.satisfied_tuples)
        reward = supp - self.supp_taus

        done = False
        if reward < 0:
            done = True

        # update state
        self.current_state = next_state
        return copy.deepcopy(self.current_state), reward, done


    def test(self, maxLHS, numREEs):
        data = []
        for step in range(numREEs):
            test_one = {}
            np.random.seed(step * 2000)
            num = np.random.choice(maxLHS, 1)[0] + 1
            np.random.seed(step * 3000)
            attrs_sc = np.random.choice(len(self.attrs), num + 1, replace=False)
            test_one['current'] = [self.attrs[e] for e in attrs_sc[:-1]]
            test_one['next'] = self.attrs[attrs_sc[-1]]
            # calculate support
            attrs_temp = [e for e in test_one['current']]
            attrs_temp += [test_one['next']]
            supp, null, null = self.cal_supp(attrs_temp, "", self.satisfied_tuples)
            if supp >= self.supp_taus:
                test_one['label'] = 1.0
            else:
                test_one['label'] = 0.0
            data.append(test_one)
        return data

    def test_np(self, maxLHS, numREEs):
        data = np.zeros((numREEs, len(self.attrs) + 1 + 1))
        for step in range(numREEs):
            test_one = {}
            np.random.seed(step * 2000)
            num = np.random.choice(maxLHS, 1)[0] + 1
            np.random.seed(step * 3000)
            attrs_sc = np.random.choice(len(self.attrs), num + 1, replace=False)
            data[step][attrs_sc[:-1]] = 1.0
            data[step][-2] = attrs_sc[-1]
            # calculate support
            attrs_temp = [self.attrs[e] for e in attrs_sc]
            supp, null, null = self.cal_supp(attrs_temp, "", self.satisfied_tuples)
            if supp >= self.supp_taus:
                data[step][-1] = 1.0
            else:
                data[step][-1] = 0.0
        return data

