import numpy as np
from selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from copent import copent
from itertools import combinations

##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 2, mode = 1):
    xyz = np.c_[x, y, z]
    yz = np.c_[y, z]
    xz = np.c_[x, z]
    return copent(xyz,k,dtype,mode) - copent(yz,k,dtype,mode) - copent(xz,k,dtype,mode)

##### estimating transfer entropy from y to x with lag [3]
def transent(x, y, lag = 1, k = 3, dtype = 2, mode = 1):
    xlen = len(x)
    ylen = len(y)
    if (xlen > ylen):
        l = ylen
    else:
        l = xlen
    if (l < (lag + k + 1)):
        return 0
    x1 = x[0:(l-lag)]
    x2 = x[lag:l]
    y = y[0:(l-lag)]
    return -ci(x2,y,x1,k,dtype,mode)


class cTE(SelectionMethod):
    def __init__(self):
        super().__init__(CTest.TE)

    
    def get_combinations(self):
        p_comb = list()
        for i in range(self.n_features):
            possible_comb = list(combinations(self.features, self.n_features-i))
            for c in possible_comb:
                p_comb.append(c)
        return p_comb

    def compute_dependencies(self):
        combs = self.get_combinations()

        for target in self.features:
            Y = self.d[target]
            
            te = dict()
            for c in range(len(combs)):
                print(c)
                X = self.d[list(combs[c])]
                te[c] = transent(Y.values, X.values, self.max_lag, k=self.n_features)
            print()

        # for t in results._single_target.keys():
        #     sel_sources = [s[0] for s in results._single_target[t]['selected_vars_sources']]
        #     if sel_sources:
        #         sel_sources_lag = [s[1] for s in results._single_target[t]['selected_vars_sources']]
        #         sel_sources_score = results._single_target[t]['selected_sources_te']
        #         sel_sources_pval = results._single_target[t]['selected_sources_pval']
        #         for s, score, pval, lag in zip(sel_sources, sel_sources_score, sel_sources_pval, sel_sources_lag):
        #             self._add_dependecies(self.features[t], self.features[s], score, pval, lag)

        return self.result