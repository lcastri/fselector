import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.CCorr import CCorr
from selection_methods.MI import MI 
from selection_methods.TE import TE


if __name__ == '__main__':   
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = pd.read_csv('sim150/interactions/interaction_1_cut.csv')
    # df = pd.read_csv('thor/interactions/interaction_10.csv')
    # df = pd.read_csv('sim/interactions/interaction_1.csv')
    # df = pd.read_csv('thor/interactions/interaction_9.csv')
    df.fillna(inplace=True, method="ffill")
    df.fillna(inplace=True, method="bfill")
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(), 
                   verbosity = CPLevel.DEBUG,
                   resfolder = 'ciao')
    
    selector_res = FS.run()
    FS.print_dependencies()
    FS.show_dependencies()

    
    

