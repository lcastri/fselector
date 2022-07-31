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


ground_truth = {
                'X_0' : [('X_1', -1), ('X_2', -1)],
                'X_1' : [],
                'X_2' : [('X_1', -1)],
                'X_3' : [('X_3', -1)],
                'X_4' : [('X_2', -1), ('X_3', -1)],
                'X_5' : [('X_4', -1), ('X_5', -1), ('X_1', -1)],
                'X_6' : [],
                }

alpha = 0.05
min_lag = 1
max_lag = 1

resfolder = 'TEvsPCMCIprova'
nsample = 1000

np.random.seed(1)
nfeature = range(3, 8)
ncoeff = 15
min_n = 0
max_n = 10
iteration = 4

RES = dict()
for n in nfeature:
    for nc in range(ncoeff):
        res_tmp = dict()
        for i in range(iteration + 1):
            res_tmp[i] = {sta._TEPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None},   
                          sta._PCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None}}
        RES[nc] = res_tmp
    
    var_names = ['X_' + str(f) for f in range(n)]
    var_names_pretty = [r'$X_' + str(f) + '$' for f in range(n)]
    gt_n = {k : ground_truth[k] for k in ground_truth.keys() if int(k[-1]) < n}
    
    d = np.random.random(size = (nsample, n))
    for nc in range(ncoeff):
        c = {i : np.random.uniform(min_n, max_n, (1, n)) for i in range(n)}
        for i in range(iteration + 1):
            coeff = deepcopy(c)
            data = deepcopy(d)
            
            if i != 0: coeff = {j: coeff[j]/np.array(10*i) for j in range(n)}    
            
            for t in range(max_lag, nsample):
                data[t, 0] += coeff[0][0][1] * data[t-1, 1] * coeff[0][0][2] * data[t-1, 2]
                data[t, 2] += coeff[2][0][1] * data[t-1, 1] ** 2
                if n > 3: data[t, 3] += data[t-1, 3] + coeff[3][0][3]
                if n > 4: data[t, 4] += coeff[4][0][2] * data[t-1, 2] - coeff[4][0][3] * data[t-1, 3]
                if n > 5: data[t, 5] += (coeff[5][0][4] * data[t-1, 4] + coeff[5][0][5] * data[t-1, 5]) / (1 + coeff[5][0][1] * data[t-1, 1])

            resdir = deepcopy(resfolder)
            resdir = resdir + '/' + str(n) + "_" + str(nc) + "_" + str(i)
    
    
            #########################################################################################################################
            # TE+PCMCI
            df = pd.DataFrame(data, columns=var_names)
            FS = FSelector(df, 
                        alpha = alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Kraskov), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
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
                
                
            RES[nc][i][sta._TEPCMCI][sta._TIME] = str(stopTE - startTE)
            RES[nc][i][sta._TEPCMCI][sta._PREC] = sta.precision(gt = gt_n, cm = new_cm_tepcmci)
            RES[nc][i][sta._TEPCMCI][sta._RECA] = sta.recall(gt = gt_n, cm = new_cm_tepcmci)
            RES[nc][i][sta._TEPCMCI][sta._F1SCORE] = sta.f1_score(RES[nc][i][sta._TEPCMCI][sta._PREC], RES[nc][i][sta._TEPCMCI][sta._RECA])


            #########################################################################################################################
            # PCMCI
            dataframe = pp.DataFrame(data, var_names = var_names_pretty)

            # init and run pcmci
            pcmci = PCMCI(dataframe = dataframe,
                        #   cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
                          cond_ind_test = CMIknn(),
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
            
            
            RES[nc][i][sta._PCMCI][sta._TIME] = str(stopPCMCI - startPCMCI)
            RES[nc][i][sta._PCMCI][sta._PREC] = sta.precision(gt = gt_n, cm = new_cm_pcmci)
            RES[nc][i][sta._PCMCI][sta._RECA] = sta.recall(gt = gt_n, cm = new_cm_pcmci)
            RES[nc][i][sta._PCMCI][sta._F1SCORE] = sta.f1_score(RES[nc][i][sta._PCMCI][sta._PREC], RES[nc][i][sta._PCMCI][sta._RECA])
            
            if not empty_cm:
                res = deepcopy(results)
                res['var_names'] = var_names_pretty
                rh.dag2(res, 
                        alpha = alpha, 
                        save_name = 'results/' + resdir + '/pcmci_dag.png', 
                        font_size = 12)
                # rh.tsdag3(res, 
                #         alpha = alpha, 
                #         save_name = 'results/' + resdir + '/pcmci_tsdag.png', 
                #         font_size = 13)
    
    res_file = open(os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json", "w+")
    json.dump(RES, res_file)
    res_file.close()
    RES.clear()
    
    sta.plot_statistics2(resfolder, n)

# sta.plot_statistics(resfolder)