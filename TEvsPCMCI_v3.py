from copy import deepcopy
import json
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.TE import TE, TEestimator
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
from datetime import datetime
import result_handler as rh
import res_statistics as sta 
import os


alpha = 0.05
min_lag = 1
max_lag = 1

resfolder = 'TEvsPCMCIlinear'
nsample = 1500

np.random.seed(1)
nfeature = range(3, 8)
ncoeff = 2
min_n = -10
max_n = 10

RES = dict()
for n in nfeature:
    res_tmp = dict()
    for nc in range(ncoeff):
        res_tmp[nc] = {sta._TEPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None},   
                       sta._PCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None}}
    
        var_names = ['X_' + str(f) for f in range(n)]
        var_names_pretty = [r'$X_' + str(f) + '$' for f in range(n)]
        

        d = np.random.random(size = (nsample, n))
        coeff = np.random.uniform(min_n, max_n, (n, n))
        gt_n = {var_names[i] : list() for i in range(len(var_names))}
        for i in range(len(var_names)):
            list_parents = list()
            for j in range(len(var_names)):
                if coeff[i][j] != 0:
                    list_parents.append((var_names[j], -1))
            gt_n[var_names[i]] = list_parents
        
        data = deepcopy(d)
        for t in range(max_lag, nsample):
            for i in range(len(var_names)):
                data[t, i] += sum([coeff[i][j]*data[t-1, j] for j in range(len(var_names))])
                    
        resdir = deepcopy(resfolder)
        resdir = resdir + '/' + str(n) + "_" + str(nc)
        
        #########################################################################################################################
        # TE+PCMCI
        df = pd.DataFrame(data, columns=var_names)
        FS = FSelector(df, 
                    alpha = alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = ParCorr(significance='analytic'),
                    verbosity = CPLevel.DEBUG,
                    resfolder = resdir)
        startTE = datetime.now()
        sel_vars = FS.run()
        stopTE = datetime.now()  
        print("\nTEPCMCI time: (hh:mm:ss.ms) {}".format(str(stopTE - startTE)))
        if sel_vars:  
            cm_tepcmci = FS.validator.val_method.return_parents_dict(FS.validator.result['graph'],
                                                                    FS.validator.result['val_matrix'])
            te_vars_dict = {sel_vars.index(v) : v for v in sel_vars}
            new_cm_tepcmci = {v: list() for v in gt_n.keys()}
            for k in cm_tepcmci.keys():
                for v in cm_tepcmci[k]:
                    new_cm_tepcmci[te_vars_dict[k]].append((te_vars_dict[v[0]], v[1])) 
        else:
            new_cm_tepcmci = {v: list() for v in gt_n.keys()}
            
            
        res_tmp[nc][sta._TEPCMCI][sta._TIME] = str(stopTE - startTE)
        res_tmp[nc][sta._TEPCMCI][sta._PREC] = sta.precision(gt = gt_n, cm = new_cm_tepcmci)
        res_tmp[nc][sta._TEPCMCI][sta._RECA] = sta.recall(gt = gt_n, cm = new_cm_tepcmci)
        res_tmp[nc][sta._TEPCMCI][sta._F1SCORE] = sta.f1_score(res_tmp[nc][sta._TEPCMCI][sta._PREC], res_tmp[nc][sta._TEPCMCI][sta._RECA])


        #########################################################################################################################
        # PCMCI
        dataframe = pp.DataFrame(data, var_names = var_names_pretty)

        # init and run pcmci
        pcmci = PCMCI(dataframe = dataframe,
                    cond_ind_test = ParCorr(significance='analytic'),
                    verbosity = 2)
                            
        startPCMCI = datetime.now()              
        results = pcmci.run_pcmci(tau_max = max_lag,
                                tau_min = min_lag,
                                pc_alpha = alpha)
        stopPCMCI = datetime.now()      
        print("\nPCMCI time: (hh:mm:ss.ms) {}".format(str(stopPCMCI - startPCMCI)))
            
        cm_pcmci = pcmci.return_parents_dict(results['graph'],
                                            results['val_matrix'])
                
        pc_vars_dict = {var_names.index(v) : v for v in var_names}
        new_cm_pcmci = {v: list() for v in var_names}
        empty_cm = True
        for k in cm_pcmci.keys():
            for v in cm_pcmci[k]:
                empty_cm = False
                new_cm_pcmci[pc_vars_dict[k]].append((pc_vars_dict[v[0]], v[1])) 
                
                
        res_tmp[nc][sta._PCMCI][sta._TIME] = str(stopPCMCI - startPCMCI)
        res_tmp[nc][sta._PCMCI][sta._PREC] = sta.precision(gt = gt_n, cm = new_cm_pcmci)
        res_tmp[nc][sta._PCMCI][sta._RECA] = sta.recall(gt = gt_n, cm = new_cm_pcmci)
        res_tmp[nc][sta._PCMCI][sta._F1SCORE] = sta.f1_score(res_tmp[nc][sta._PCMCI][sta._PREC], res_tmp[nc][sta._PCMCI][sta._RECA])
                
        if not empty_cm:
            res = deepcopy(results)
            res['var_names'] = var_names_pretty
            rh.dag2(res, 
                    alpha = alpha, 
                    save_name = 'results/' + resdir + '/pcmci_dag.png', 
                    font_size = 12)
    RES[n] = res_tmp    
    res_file = open(os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json", "w+")
    json.dump(res_tmp, res_file)
    res_file.close()
    res_tmp.clear()    
    
    # sta.plot_statistics2(resfolder, n)

# sta.plot_statistics(resfolder)