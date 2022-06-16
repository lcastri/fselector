from abc import ABC, abstractmethod
from CPrinter import CP

class SelectionMethod(ABC):
    def __init__(self, name):
        self.name = name
        self.d = None
        self.alpha = None
        self.lag = None
        self.result = dict()


    @property
    def sm_name(self):
        """
        Returns Selection Method name

        Returns:
            str: Selection Method name
        """
        return self.name.value

    
    @property
    def features(self):
        """
        Returns list of features

        Returns:
            list(str): list of feature names
        """
        return list(self.d.columns.values)


    @property
    def n_samples(self):
        """
        Returns number of samples in the timeseries

        Returns:
            int: number of sample in the timeseries
        """
        return self.d.shape[0]


    @property
    def n_features(self):
        """
        Returns the number of feature (process) contained in the DataFrame

        Returns:
            int: number of feature (process) contained in the DataFrame
        """
        return self.d.shape[1]


    def initialise(self, d, alpha, lag):
        """
        Initialises the selection method

        Args:
            d (DataFrame): dataframe
            alpha (float): significance threshold
            lag (int): lag time
        """
        self.d = d
        self.alpha = alpha
        self.lag = lag


    def prepare_ts(self, target, apply_lag = True):
        """
        prepare the dataframe to analysis

        Args:
            d (DataFrame): original dataframe
            target (str): name target var

        Returns:
            tuple(DataFrame): X and target DataFrames
        """
        if apply_lag:
            Y = self.d[target][self.lag:]
            X = self.d.loc[:, self.d.columns != target][:-self.lag]
        else:
            Y = self.d[target]
            X = self.d.loc[:, self.d.columns != target]
        return X, Y


    def add_dependecies(self, t, s, score, pval, lag):
        """
        Adds found dependency from source (s) to target (t) specifying the 
        score, pval and the lag

        Args:
            t (str): target feature name
            s (str): source feature name
            score (float): selection method score
            pval (float): pval associated to the dependency
            lag (int): lag time of the dependency
        """
        if t not in self.result: self.result[t] = list()
        self.result[t].append((s, score, pval, lag))


    @abstractmethod
    def compute_dependencies(self) -> dict:
        pass