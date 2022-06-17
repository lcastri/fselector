import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.SelectionMethod import CCorr, MI, TE


if __name__ == '__main__':

    df = pd.read_csv('interaction_1.csv')
    FS = FSelector(df, 
                   alpha = 0.05, 
                   min_lag = 1, 
                   max_lag = 1, 
                   sel_method = TE(), 
                   verbosity = CPLevel.DEBUG)
    FS.run()
                        
