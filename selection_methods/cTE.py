import numpy as np
from selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from copent import copent
from itertools import combinations
from pyunicorn.timeseries import Surrogates
from PyIF.te_compute import te_compute


##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 2, mode = 1):
    xyz = np.c_[x, y, z]
    yz = np.c_[y, z]
    xz = np.c_[x, z]
    return copent(xyz,k,dtype,mode) - copent(yz,k,dtype,mode) - copent(xz,k,dtype,mode)

##### estimating transfer entropy from y to x with lag [3]
def transent(target, source, lag = 1, k = 3, dtype = 2, mode = 1):
    xlen = len(target)
    ylen = len(source)
    if (xlen > ylen):
        l = ylen
    else:
        l = xlen
    if (l < (lag + k + 1)):
        return 0
    x1 = target[0:(l-lag)]
    x2 = target[lag:l]
    source = source[0:(l-lag)]
    return abs(ci(x2,source,x1,k,dtype,mode))


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
        for target in self.features:
            print(target)
            X, Y = self._prepare_ts(target, self.max_lag, apply_lag=False)
            for source in X:
                te = abs(te_compute(X[source].values, Y.values, k = 3, embedding=1, safetyCheck=False, GPU=False))

                null_dist = np.zeros(100)
                orig_data = np.c_[X[source].values, Y.values]
                s = Surrogates(original_data=orig_data, silence_level=3)
                for i in range(len(null_dist)):
                    surrogates = s.AAFT_surrogates(original_data=orig_data)
                    s.clear_cache()
                    s._fft_cached=False
                    null_dist[i] = abs(te_compute(surrogates[:,0], surrogates[:,1], k=1, embedding=1, safetyCheck=False, GPU=False))

                pval = sum(null_dist > te)/len(null_dist)
                self._add_dependecies(target, source, te, pval, self.max_lag)

        return self.result