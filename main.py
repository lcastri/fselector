import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.CCorr import CCorr
from selection_methods.MI import MI 
from selection_methods.TE import TE
from selection_methods.cTE import cTE


if __name__ == '__main__':
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    # df = pd.read_csv('sim/interactions/interaction_1.csv')
    df = pd.read_csv('sim150/interactions/interaction_1_cut.csv')
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(), 
                   verbosity = CPLevel.DEBUG,
                   resfolder = 'sim150')
    
    selector_res = FS.run()
    FS.print_dependencies()
    FS.show_dependencies()

    
    

