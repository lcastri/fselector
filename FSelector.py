import copy
from datetime import datetime
from selection_methods.SelectionMethod import SelectionMethod
from CPrinter import CPLevel, CP
import matplotlib.pyplot as plt
import numpy as np
from selection_methods.constants import *
from utilities import utilities as utils, logger as log
from FValidator import FValidator
from tigramite.independence_tests import CondIndTest
import sys

import warnings
warnings.filterwarnings('ignore')


class FSelector():

    def __init__(self, d, alpha, min_lag, max_lag, sel_method: SelectionMethod, val_condtest: CondIndTest, verbosity: CPLevel, resfolder = None):
        self.o_d = d
        self.o_dependencies = None
        
        self.d = d
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.sel_method = sel_method
        self.dependencies = None
        self.result = None
        self.score_threshold = utils.Thres.INIT

        self.dependency_path = None
        if resfolder is not None:
            utils.create_results_folder()
            logpath, self.dependency_path = utils.get_selectorpath(resfolder)
            sys.stdout = log.Logger(logpath)
        
        self.validator = FValidator(d, alpha, min_lag, max_lag, val_condtest, resfolder, verbosity)       
        CP.set_verbosity(verbosity)


# region PROPERTIES

    @property
    def o_features(self):
        """
        Returns list of original features (no filtered)

        Returns:
            list(str): list of feature names
        """
        return list(self.o_d.columns.values)


    @property
    def o_pretty_features(self):
        """
        Returns list of original features (no filtered) with LATEX symbols

        Returns:
            list(str): list of feature names
        """
        return [r'$' + str(column) + '$' for column in self.o_d.columns.values]


    @property
    def o_nfeatures(self):
        """
        Number of original features (no filtered)

        Returns:
            int: number of features
        """
        return len(self.o_d.columns.values)


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

# region PUBLIC

    def run_selector(self):
        """
        Run selection method
        """
        CP.info("\n")
        CP.info(utils.DASH)
        CP.info("Selecting relevant features among: " + str(self.features))
        CP.info("Selection method: " + self.sel_method.name)
        CP.info("Significance level: " + str(self.alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))

        self.sel_method.initialise(self.d, self.alpha, self.min_lag, self.max_lag)
        self.dependencies = self.sel_method.compute_dependencies()
        self.o_dependecies = copy.deepcopy(self.dependencies)

    
    def run(self):
        """
        Run Selector and Validator
        
        Returns:
            list(str): list of selected variable names
        """
        
        self.run_selector()        
        while self.score_threshold is not utils.Thres.NOFOUND:
            # exclude dependencies based on score threshold
            self.__apply_score_threshold()
            
            # list of selected features based on dependencies
            sel_features = self.get_selected_features()
            if not sel_features: #FIXME: to check if it is empty
                break

            # shrink dataframe d and dependencies by sel_features
            self.__shrink_d(sel_features)

            # selected links to check by the validator
            selected_links = self.__get_selected_links()
            
            # causal model on selected links
            self.validator.d = self.d
            pcmci_result = self.validator.run(selected_links)
            
            if CP.verbosity.value >= CPLevel.DEBUG.value:
                # comparison between obtained causal model and selected links 
                self.__compare_selected_links(pcmci_result)
                
            # get score threshold based on causal model
            self.__get_score_threshold(pcmci_result)
        
        if self.get_selected_features(): #FIXME: to check if it is empty
            self.validator.save_result()
            # self.validator.build_dag()
            # self.validator.build_ts_dag()
            self.validator.create_plot()
        return self.get_selected_features()


    def get_selected_features(self):
        """
        Defines the list of selected variables for d

        Returns:
            list(str): list of selected variable names
        """
        f_list = list()
        for t in self.dependencies:
            sources_t = self.__get_dependencies_for_target(t)
            if sources_t:
                sources_t.append(t)
            f_list = list(set(f_list + sources_t))
        self.result = [f for f in self.features if f in f_list]
        CP.info("\nFeature selected: "+ str(self.result))

        return self.result
    
    
    def show_dependencies(self):
        """
        Saves dependencies graph if resfolder is set otherwise it shows the figure
        """
        # FIXME: LAG not considered
        dependencies_matrix = self.__get_dependencies_matrix()

        fig, ax = plt.subplots()
        im = ax.imshow(dependencies_matrix, cmap=plt.cm.Greens, interpolation='nearest', vmin=0, vmax=1, origin='lower')
        fig.colorbar(im, orientation='vertical', label="score")

        plt.xlabel("Sources")
        plt.ylabel("Targets")
        plt.xticks(ticks = range(0, self.o_nfeatures), labels = self.o_pretty_features, fontsize = 8)
        plt.yticks(ticks = range(0, self.o_nfeatures), labels = self.o_pretty_features, fontsize = 8)
        plt.title("Dependencies")

        if self.dependency_path is not None:
            plt.savefig(self.dependency_path, dpi=300)
        else:
            plt.show()


    def print_dependencies(self):
        """
        Print dependencies found by the selector
        """
        for t in self.o_dependecies:
            print()
            print()
            print(utils.DASH)
            print("Target", t)
            print(utils.DASH)
            print('{:<10s}{:>15s}{:>15s}{:>15s}'.format('SOURCE', 'SCORE', 'PVAL', 'LAG'))
            print(utils.DASH)
            for s in self.o_dependecies[t]:
                print('{:<10s}{:>15.3f}{:>15.3f}{:>15d}'.format(s[SOURCE], s[SCORE], s[PVAL], s[LAG]))       

# endregion

# region PRIVATE

    def __shrink_d(self, selected_features):
        """
        Shrinks dataframe d and dependencies based on the selected features

        Args:
            selected_features (list(str)): features selected by the selector
        """
        self.d = self.d[selected_features]
        difference_set = self.dependencies.keys() - self.features
        for d in difference_set: del self.dependencies[d]


    def __get_dependencies_for_target(self, t):
        """
        Returns list of sources for a specified target

        Args:
            t (str): target variable name

        Returns:
            list(str): list of sources for target t
        """
        return [s[SOURCE] for s in self.dependencies[t]]


    def __get_dependencies_matrix(self):
        """
        Returns a matrix composed by scores for each target

        Returns:
            np.array: score matrix
        """
        dep_mat = list()
        for t in self.o_dependecies:
            dep_vet = [0] * self.o_nfeatures
            for s in self.o_dependecies[t]:
                dep_vet[self.o_features.index(s[SOURCE])] = s[SCORE]
            dep_mat.append(dep_vet)

        dep_mat = np.array(dep_mat)
        inf_mask = np.isinf(dep_mat)
        neginf_mask = np.isneginf(dep_mat)
        max_dep_mat = np.max(dep_mat[(dep_mat != -np.inf) & (dep_mat != np.inf)])
        min_dep_mat = np.min(dep_mat[(dep_mat != -np.inf) & (dep_mat != np.inf)])

        dep_mat[inf_mask] = max_dep_mat
        dep_mat[neginf_mask] = min_dep_mat
        dep_mat = (dep_mat - min_dep_mat) / (max_dep_mat - min_dep_mat)
        return dep_mat


    def __get_selected_links(self):
        """
        Return selected links found by the selector
        in this form: {0: [(0,-1), (2,-1)]}

        Returns:
            dict: selected links
        """
        # FIXME: find a way to compute autocorrelation instead of having it as default
        sel_links = {self.features.index(f):list() for f in self.features}
        for t in self.dependencies:

            # add autocorrelation links
            for lag in range(self.min_lag, self.max_lag + 1):
                sel_links[self.features.index(t)].append((self.features.index(t), -lag))
            
            # add external links
            for s in self.dependencies[t]:
                sel_links[self.features.index(t)].append((self.features.index(s[SOURCE]), -s[LAG]))

        return sel_links
    
    
    def __apply_score_threshold(self):
        """
        Exclude dependencies based on score threshold found
        """
        if (self.score_threshold is not utils.Thres.NOFOUND) and (self.score_threshold != utils.Thres.INIT):
            CP.debug(utils.DASH)
            CP.debug("Applying score threshold")
            depend = copy.deepcopy(self.dependencies)
            for t in depend:
                sources = depend[t]
                for s in sources:
                    if s[SCORE] <= self.score_threshold:
                        CP.debug("Removing source " + s[SOURCE] + " from target " + t + " : " + str(s[SCORE]))
                        self.dependencies[t].remove(s)      
            
        
    
    def __get_score_threshold(self, causal_model):
        """
        Calculate score threshold based on the causal model 
        obtained by the causal discovery analysis

        Args:
            causal_model (dict): causal model described as dictionary

        Returns:
            float: score threshold
        """
        CP.debug(utils.DASH)
        CP.debug("Difference(s)")
        CP.debug(utils.DASH)
        score_removed = list()
        for t in self.dependencies:
            for s in self.dependencies[t]:
                if (self.features.index(s[SOURCE]), -s[LAG]) not in causal_model[self.features.index(t)]:
                    CP.debug("Source " + s[SOURCE] + " for target " + t)
                    score_removed.append(s[SCORE])
        if score_removed:
            CP.info("\n==> Difference(s) between dependencies and causal model")
            CP.info("==> Stopping criteria NOT REACHED")
            self.score_threshold = max(score_removed)
            CP.debug("Score threshold = " + str(self.score_threshold))
        else:
            CP.debug("None")
            CP.info("\n==> NO difference(s) between dependencies and causal model")
            CP.info("==> Stopping criteria REACHED")
            self.score_threshold = utils.Thres.NOFOUND
        return self.score_threshold
        
        
    def __compare_selected_links(self, validator_sel_links):
        selector_sel_links = self.__get_selected_links()
        print()
        print(utils.DASH)
        print("Selector selected links:")
        print(utils.DASH)
        for t in selector_sel_links.keys():
            print("Sources for target", self.features[t], ":", [self.features[s[0]] for s in selector_sel_links[t]])
            
        print(utils.DASH)
        print("Validator selected links:")
        print(utils.DASH)
        for t in validator_sel_links.keys():
            print("Sources for target", self.features[t], ":", [self.features[s[0]] for s in validator_sel_links[t]])
            
# endregion
