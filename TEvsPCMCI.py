from copy import deepcopy
import json
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.TE import TE, TEestimator
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
from datetime import datetime
import result_handler as rh
import res_statistics as sta 
import os


ground_truth = {
                'X_0' : [],
                'X_1' : [('X_0', -1)],
                'X_2' : [('X_0', -1)],
                'X_3' : [('X_0', -1), ('X_1', -1)],
                'X_4' : [('X_3', -1), ('X_5', -1)],
                'X_5' : [('X_1', -1), ('X_2', -1)],
                }


alpha = 0.05
min_lag = 1
max_lag = 1

resfolder = 'TEvsPCMCI2'
np.random.seed(1)
nsample = 1500
nfeature = 6
d = np.random.randn(nsample, nfeature)

min_n = 5
max_n = 10
c = dict()
c[0] = np.array([np.random.uniform(min_n, max_n)])
c[1] = np.array([np.random.uniform(min_n, max_n)])
c[2] = np.array([np.random.uniform(min_n, max_n)])
c[3] = np.array([np.random.uniform(min_n, max_n)])
c[4] = np.array([np.random.uniform(min_n, max_n), np.random.uniform(min_n, max_n)])
c[5] = np.array([np.random.uniform(min_n, max_n)])

var_names = ['X_' + str(f) for f in range(nfeature)]
var_names_pretty = [r'$X_' + str(f) + '$' for f in range(nfeature)]

iteration = range(20)
RES = {i : {sta._TEPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None}, 
            sta._PCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None}} for i in iteration}

for i in iteration:
    coeff = deepcopy(c)
    data = deepcopy(d)
    if i != 0:
        coeff = {n: coeff[n]/np.array(2*i) for n in range(nfeature)}    
    for t in range(max_lag, nsample):
        data[t, 1] += coeff[1][0] * data[t-1, 0]
        data[t, 2] += coeff[2][0] * data[t-1, 0]**2
        data[t, 3] += coeff[3][0] * data[t-1, 0] * data[t-1, 1]
        data[t, 4] += coeff[4][0] + data[t-1, 3] + coeff[4][1] * data[t-1, 5]
        data[t, 5] += coeff[5][0] + data[t-1, 1] * data[t-1, 2]

    resdir = deepcopy(resfolder)
    resdir = resdir + '/' + str(i)
    
    
    #########################################################################################################################
    # TE+PCMCI
    df = pd.DataFrame(data, columns=var_names)
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(TEestimator.Gaussian), 
                   val_condtest = GPDC(significance = 'analytic', gp_params = None),
                   verbosity = CPLevel.DEBUG,
                   resfolder = resdir)

    startTE = datetime.now()
    sel_vars = FS.run()
    stopTE = datetime.now()  
    if sel_vars:  
        cm_tepcmci = FS.validator.val_method.return_parents_dict(FS.validator.result['graph'],
                                                                FS.validator.result['val_matrix'])
        te_vars_dict = {sel_vars.index(v) : v for v in sel_vars}
        new_cm_tepcmci = {v: list() for v in ground_truth.keys()}
        for k in cm_tepcmci.keys():
            for v in cm_tepcmci[k]:
                new_cm_tepcmci[te_vars_dict[k]].append((te_vars_dict[v[0]], v[1])) 
    else:
        new_cm_tepcmci = {v: list() for v in ground_truth.keys()}
        
        
    RES[i][sta._TEPCMCI][sta._TIME] = str(stopTE - startTE)
    RES[i][sta._TEPCMCI][sta._PREC] = sta.precision(gt = ground_truth, cm = new_cm_tepcmci)
    RES[i][sta._TEPCMCI][sta._RECA] = sta.recall(gt = ground_truth, cm = new_cm_tepcmci)
    RES[i][sta._TEPCMCI][sta._F1SCORE] = sta.f1_score(RES[i][sta._TEPCMCI][sta._PREC], RES[i][sta._TEPCMCI][sta._RECA])

    #########################################################################################################################
    # PCMCI
    dataframe = pp.DataFrame(data, var_names = var_names_pretty)

    # init and run pcmci
    pcmci = PCMCI(dataframe = dataframe,
                  cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
                  verbosity = 2)
                
    startPCMCI = datetime.now()              
    results = pcmci.run_pcmci(tau_max = max_lag,
                              tau_min = min_lag,
                              pc_alpha = alpha)
    stopPCMCI = datetime.now()              
    cm_pcmci = pcmci.return_parents_dict(results['graph'],
                                         results['val_matrix'])
    
    pc_vars_dict = {var_names.index(v) : v for v in var_names}
    new_cm_pcmci = {v: list() for v in var_names}
    empty_cm = True
    for k in cm_pcmci.keys():
        for v in cm_pcmci[k]:
            empty_cm = False
            new_cm_pcmci[pc_vars_dict[k]].append((pc_vars_dict[v[0]], v[1])) 
    
    
    RES[i][sta._PCMCI][sta._TIME] = str(stopPCMCI - startPCMCI)
    RES[i][sta._PCMCI][sta._PREC] = sta.precision(gt = ground_truth, cm = new_cm_pcmci)
    RES[i][sta._PCMCI][sta._RECA] = sta.recall(gt = ground_truth, cm = new_cm_pcmci)
    RES[i][sta._PCMCI][sta._F1SCORE] = sta.f1_score(RES[i][sta._PCMCI][sta._PREC], RES[i][sta._PCMCI][sta._RECA])
    
    if not empty_cm:
        res = deepcopy(results)
        res['var_names'] = var_names_pretty
        rh.dag(res, 
            alpha = alpha, 
            save_name = 'results/' + resdir + '/pcmci_dag.png', 
            font_size = 18)
    
res_file = open(os.getcwd() + "/results/"+resfolder + "/res.json", "w+")
json.dump(RES, res_file)
res_file.close()

sta.plot_statistics(resfolder)