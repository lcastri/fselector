from selection_methods.SelectionMethod import SelectionMethod
from CPrinter import CPLevel, CP
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class FSelector():

    def __init__(self, d, alpha, lag, sel_method: SelectionMethod, verbosity: CPLevel):
        self.d = d
        self.alpha = alpha
        self.lag = lag
        self.sel_method = sel_method
        self.dependencies = None
        self.result = None

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


    def get_dependencies_for_target(self, t):
        """
        Returns list of sources for a specified target

        Args:
            t (str): target variable name

        Returns:
            list(str): list of sources for target t
        """
        return [s[0] for s in self.dependencies[t]]


    def get_dep_score_for_target(self, t):
        """
        Returns list of scores for a specified target

        Args:
            t (str): target variable name

        Returns:
            list(float): list of scores for target t
        """
        return [(s[0], s[1]) for s in self.dependencies[t]]


    def __get_selected_features(self):
        """
        Returns the list of selected variables for d

        Returns:
            list(str): list of selected variable names
        """
        f_list = list()
        for t in self.dependencies:
            sources_t = self.get_dependencies_for_target(t)
            if sources_t:
                sources_t.append(t)
            f_list = list(set(f_list + sources_t))
        self.result = f_list

    
    def run_selector(self):
        """
        Run selection method
        """
        CP.info("\n")
        CP.info("Selecting relevant features among: " + str(self.features))
        CP.info("Selection method: " + self.sel_method.name.value)
        CP.info("Significance level: " + str(self.alpha))

        self.sel_method.initialise(self.d, self.alpha, self.lag)
        self.dependencies = self.sel_method.compute_dependencies()
        self.__get_selected_features()


    def run(self):
        """
        Run Feature Selector, print and show results

        Wrapper for 3 methods:
        - run_selector
        - print_dependencies
        - show_dependencies
        """
        self.run_selector()
        self.print_dependencies()
        self.show_dependencies()


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
                dep_vet[self.features.index(s[0])] = s[1]
            dep_mat.append(dep_vet)

        dep_mat = np.array(dep_mat)
        dep_mat = (dep_mat - np.min(dep_mat)) / (np.max(dep_mat) - np.min(dep_mat))
        return dep_mat


    def show_dependencies(self):
        # FIXME: LAG not considered
        # FIXME: add colormap on the right
        dependencies_matrix = self.__get_dependencies_matrix()

        fig, ax = plt.subplots()
        im = ax.imshow(dependencies_matrix, cmap=plt.cm.Greens, interpolation='nearest', vmin=0, vmax=1, origin='lower')
        fig.colorbar(im, orientation='vertical', label="score")

        plt.xlabel("Sources")
        plt.ylabel("Targets")
        plt.xticks(ticks = range(0, self.nfeatures), labels = self.pretty_features, fontsize = 8)
        plt.yticks(ticks = range(0, self.nfeatures), labels = self.pretty_features, fontsize = 8)
        plt.title("Dependencies")
        plt.show()


    def print_dependencies(self):
        print()
        for t in self.dependencies:
            print(t, "dependencies:")
            for s in self.dependencies[t]:
                print("\t - %s \t score: %.3f \t pval: %.3f \t lag: %2d" % (s[0], s[1], s[2], s[3]))
            print("--------------------------------------------------")

        
