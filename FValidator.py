import copy
import pickle
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
import tigramite.data_processing as pp
import tigramite.plotting as tigraplot
import numpy as np
from CPrinter import CPLevel, CP
from utilities import utilities as utils, dag


class FValidator():
    def __init__(self, d, alpha, min_lag, max_lag, resfolder, verbosity: CPLevel):
        self.d = d
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = None
        self.val_method = None
        self.verbosity = verbosity.value

        self.respath = None
        self.dag_path = None
        self.ts_dag_path = None
        if resfolder is not None:
            self.respath, self.dag_path, self.ts_dag_path = utils.get_validatorpaths(resfolder)
        
        
# region PROPERTIES

    @property
    def features(self):
        """
        Returns list of features

        Returns:
            list(str): list of feature names
        """
        return list(self.d.columns.values)


    @property
    def pretty_features(self):
        """
        Returns list of features with LATEX symbols

        Returns:
            list(str): list of feature names
        """
        return [r'$' + str(column) + '$' for column in self.d.columns.values]
    

    @property
    def nfeatures(self):
        """
        Number of features

        Returns:
            int: number of features
        """
        return len(self.d.columns.values)
    
# endregion    
        
        
    def run(self, selected_links):
        """
        Run causal discovery algorithm to find a score threshold

        Returns:
            dict: estimated causal model
        """
        CP.info('\n')
        CP.info(utils.DASH)
        CP.info(utils.bold("Running Causal Discovery Algorithm to find score threshold"))

        # build tigramite dataset
        vector = np.vectorize(float)
        sub_data = vector(self.d)
        dataframe = pp.DataFrame(data = sub_data,
                                 var_names = self.pretty_features)
        
        
        # init and run pcmci
        self.val_method = PCMCI(dataframe = dataframe,
                                cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
                                verbosity = self.verbosity)

        self.result = self.val_method.run_pcmci(selected_links = selected_links,
                                                tau_max = self.max_lag,
                                                tau_min = self.min_lag,
                                                pc_alpha = self.alpha)
        
        
        return self.__return_parents_dict()
    
    
    def build_ts_dag(self):
        """
        Saves timeseries dag plot if resfolder is set otherwise it shows the figure
        """
        tigraplot.plot_time_series_graph(figsize = (6, 4),
                                         graph = self.result['graph'],
                                         val_matrix = self.result['val_matrix'],
                                         var_names = self.pretty_features,
                                         link_colorbar_label = 'MCI',
                                         save_name = self.ts_dag_path,
                                         edge_ticks = 0.2,
                                         vmin_edges = 0.,
                                         vmax_edges = round(self.result['val_matrix'].max(), 1),
                                         cmap_edges = 'OrRd',
                                         arrow_linewidth = 3)


    def build_dag(self):
        """
        Saves dag plot if resfolder is set otherwise it shows the figure
        """
        dag.plot_graph(figsize = (6, 4), 
                       graph = self.result['graph'],
                       val_matrix = self.result['val_matrix'],
                       var_names = self.pretty_features,
                       link_colorbar_label='cross-MCI',
                       node_colorbar_label='auto-MCI',
                       save_name = self.dag_path,
                       edge_ticks = 0.2,
                       vmin_edges = 0.,
                       vmax_edges = round(self.result['val_matrix'].max(), 1),
                       cmap_edges = 'OrRd',
                       link_label_fontsize= 0,
                       arrow_linewidth = 3)
        
        
    def save_result(self):
        """
        Save causal discovery results as pickle file if resfolder is set
        """
        if self.respath is not None:
            res = copy.deepcopy(self.result)
            res['alpha'] = self.alpha
            res['var_names'] = self.pretty_features
            res['dag_path'] = self.dag_path
            res['ts_dag_path'] = self.ts_dag_path
            with open(self.respath, 'wb') as resfile:
                pickle.dump(res, resfile)
        
        
    def __return_parents_dict(self):
        """
        Returns dictionary of parents sorted by val_matrix filtered by alpha

        Returns:
            dict: Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.
        """
        graph = self.result['graph']
        val_matrix = self.result['val_matrix']
        p_matrix = self.result['p_matrix']
        
        # Initialize the return value
        parents_dict = dict()
        for j in range(self.nfeatures):
            # Get the good links
            good_links = np.argwhere(graph[:, j, 1:] == "-->")
            # Build a dictionary from these links to their values
            links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1]) 
                     for i, tau in good_links if p_matrix[i, j, abs(tau) + 1] <= self.alpha}
            # Sort by value
            parents_dict[j] = sorted(links, key=links.get, reverse=True)
        
        return parents_dict
