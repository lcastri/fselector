import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.CCorr import CCorr
from selection_methods.MI import MI 
from selection_methods.TE import TE

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
import tigramite.data_processing as pp
import numpy as np


def return_parents_dict(result, alpha):
    """Returns dictionary of parents sorted by val_matrix.
    If parents are unclear (link with o or x), then no parent 
    is returned. 
    Parameters
    ----------
    graph : array of shape [N, N, tau_max+1]
        Causal graph, see description above for interpretation.
    val_matrix : array-like
        Matrix of test statistic values. Must be of shape (N, N, tau_max +
        1).
    include_lagzero_parents : bool (default: False)
        Whether the dictionary should also return parents at lag
        zero. 
    Returns
    -------
    parents_dict : dict
        Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
        containing estimated parents.
    """
    graph = result['graph']
    val_matrix = result['val_matrix']
    p_matrix = result['p_matrix']
    
    # Initialize the return value
    parents_dict = dict()
    for j in range(len(result['graph'])):
        # Get the good links
        good_links = np.argwhere(graph[:, j, 1:] == "-->")
        # Build a dictionary from these links to their values
        links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1]) for i, tau in good_links if p_matrix[i, j, abs(tau) + 1] < alpha}
        # Sort by value
        parents_dict[j] = sorted(links, key=links.get, reverse=True)
    
    return parents_dict


if __name__ == '__main__':
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = pd.read_csv('interaction_1_cut.csv')
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(), 
                   verbosity = CPLevel.DEBUG)
    
    selector_res = FS.run()
    # FS.print_dependencies()
    # FS.show_dependencies()


    while FS.score_threshold is not None:
        FS.apply_score_threshold()
        FS.get_selected_features()
        selected_links = FS.get_selected_links()
        
        # build tigramite dataset
        vector = np.vectorize(float)
        sub_data = vector(df)
        dataframe = pp.DataFrame(data = sub_data,
                                var_names = FS.pretty_features)
        
        # init and run pcmci
        pcmci = PCMCI(dataframe = dataframe,
                    cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = 2)
        
        
        result = pcmci.run_pcmci(selected_links = selected_links,
                                tau_max = max_lag,
                                tau_min = min_lag,
                                pc_alpha = alpha)
        
        
        pcmci_result = return_parents_dict(result, alpha)
        
        te_thres = FS.get_score_threshold(pcmci_result)
        print("LOOP COMPLETED")
        
    FS.get_selected_features()

    
    

