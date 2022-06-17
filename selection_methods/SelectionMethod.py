from abc import ABC, abstractmethod
from enum import Enum
from CPrinter import CP
from CCorr import CCorr
from MI import MI
from TE import TE

class CTest(Enum):
    CCorr = "Cross-correlation"
    MI = "Mutual Information"
    HSICLasso = "HSICLasso"
    TE = "Transfer Entropy"
    MIT = "Momentary Information Transfer"


class SelectionMethod(ABC):
    def __init__(self, ctest):
        self.ctest = ctest
        self.d = None
        self.alpha = None
        self.min_lag = None
        self.max_lag = None
        self.result = dict()


    @property
    def name(self):
        """
        Returns Selection Method name

        Returns:
            str: Selection Method name
        """
        return self.ctest.value

    
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


    def initialise(self, d, alpha, min_lag, max_lag):
        """
        Initialises the selection method

        Args:
            d (DataFrame): dataframe
            alpha (float): significance threshold
            min_lag (int): min lag time
            max_lag (int): max lag time
        """
        self.d = d
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = {f:list() for f in self.features}


    @abstractmethod
    def compute_dependencies(self) -> dict:
        pass


    def _prepare_ts(self, target, lag, apply_lag = True):
        """
        prepare the dataframe to the analysis

        Args:
            target (str): name target var
            lag (int): lag time to apply
            apply_lag (bool, optional): True if you want to apply the lag, False otherwise. Defaults to True.

        Returns:
            tuple(DataFrame, DataFrame): source and target dataframe
        """
        if apply_lag:
            Y = self.d[target][lag:]
            X = self.d.loc[:, self.d.columns != target][:-lag]
        else:
            Y = self.d[target]
            X = self.d.loc[:, self.d.columns != target]
        return X, Y


    def _add_dependecies(self, t, s, score, pval, lag):
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
        self.result[t].append((s, score, pval, lag))