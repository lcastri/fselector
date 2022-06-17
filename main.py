import pandas as pd
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.CCorr import CCorr
from selection_methods.MI import MI 
from selection_methods.TE import TE


if __name__ == '__main__':

    df = pd.read_csv('interaction_2.csv')
    FS = FSelector(df, 
                   alpha = 0.05, 
                   min_lag = 1, 
                   max_lag = 1, 
                   sel_method = TE(), 
                   verbosity = CPLevel.INFO)
    FS.run()
    # FS.print_dependencies()
    # FS.show_dependencies()

    selected_links = FS.get_selected_links()
    

