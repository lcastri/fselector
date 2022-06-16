import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.CCorr import CCorr


if __name__ == '__main__':

    df = pd.read_csv('interaction_1.csv')
    FS = FSelector(df, 
                   alpha = 0.05, 
                   lag = 1, 
                   sel_method = CCorr(), 
                   verbosity = CPLevel.DEBUG)
    FS.run()
                        
