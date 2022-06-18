import copy
from selection_methods.SelectionMethod import SelectionMethod
from CPrinter import CPLevel, CP
import matplotlib.pyplot as plt
import numpy as np

from selection_methods.constants import *


class FSelector():

    def __init__(self, d, alpha, min_lag, max_lag, sel_method: SelectionMethod, verbosity: CPLevel):
        self.d = d
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.sel_method = sel_method
        self.dependencies = None
        self.result = None
        self.score_threshold = -1

        CP.set_verbosity(verbosity)


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


    def run(self):
        """
        Run selection method
        """
        CP.info("\n")
        CP.info("Selecting relevant features among: " + str(self.features))
        CP.info("Selection method: " + self.sel_method.name)
        CP.info("Significance level: " + str(self.alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))

        self.sel_method.initialise(self.d, self.alpha, self.min_lag, self.max_lag)
        self.dependencies = self.sel_method.compute_dependencies()
        return self.get_selected_features()


    def get_selected_features(self):
        """
        Defines the list of selected variables for d
        """
        f_list = list()
        for t in self.dependencies:
            sources_t = self.__get_dependencies_for_target(t)
            if sources_t:
                sources_t.append(t)
            f_list = list(set(f_list + sources_t))
        self.result = f_list
        CP.info("Feature selected: "+ str(self.result))

        return self.result


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
        for t in self.dependencies:
            dep_vet = [0] * self.nfeatures
            for s in self.dependencies[t]:
                dep_vet[self.features.index(s[SOURCE])] = s[SCORE]
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


    def get_selected_links(self):
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
    
    
    def apply_score_threshold(self):
        if (self.score_threshold is not None) and (self.score_threshold != -1):
            CP.debug("Score threshold = " + str(self.score_threshold))
            depend = copy.deepcopy(self.dependencies)
            for t in depend:
                sources = depend[t]
                for s in sources:
                    if s[SCORE] <= self.score_threshold:
                        CP.debug("Removing source " + s[SOURCE] + " from target " + t + " : " + str(s[SCORE]))
                        self.dependencies[t].remove(s)      
        print()             
        
    
    def get_score_threshold(self, causal_model):
        """
        Calculate score threshold based on the causal model 
        obtained by the causal discovery analysis

        Args:
            causal_model (dict): causal model described as dictionary

        Returns:
            float: score threshold
        """
        score_removed = list()
        for t in self.dependencies:
            for s in self.dependencies[t]:
                if (self.features.index(s[SOURCE]), -s[LAG]) not in causal_model[self.features.index(t)]:
                    score_removed.append(s[SCORE])
        if score_removed:
            CP.debug("score removed \n" + '\n'.join([str(te) for te in score_removed]))
            CP.debug("score threshold " + str(max(score_removed)))
            self.score_threshold = max(score_removed)
        else:
            self.score_threshold = None
        return self.score_threshold


    def show_dependencies(self):
        """
        Plot dependencies graph
        """
        # FIXME: LAG not considered
        dependencies_matrix = self.__get_dependencies_matrix()

        fig, ax = plt.subplots()
        im = ax.imshow(dependencies_matrix, cmap=plt.cm.Wistia, interpolation='nearest', vmin=0, vmax=1, origin='lower')
        fig.colorbar(im, orientation='vertical', label="score")

        plt.xlabel("Sources")
        plt.ylabel("Targets")
        plt.xticks(ticks = range(0, self.nfeatures), labels = self.pretty_features, fontsize = 8)
        plt.yticks(ticks = range(0, self.nfeatures), labels = self.pretty_features, fontsize = 8)
        plt.title("Dependencies")
        plt.show()


    def print_dependencies(self):
        """
        Print dependencies found by the selector
        """
        dash = '-' * 55
        for t in self.dependencies:
            print()
            print()
            print(dash)
            print("Target", t)
            print(dash)
            print('{:<10s}{:>15s}{:>15s}{:>15s}'.format('SOURCE', 'SCORE', 'PVAL', 'LAG'))
            print(dash)
            for s in self.dependencies[t]:
                print('{:<10s}{:>15.3f}{:>15.3f}{:>15d}'.format(s[SOURCE], s[SCORE], s[PVAL], s[LAG]))
        
