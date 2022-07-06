from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
import tigramite.plotting as tigraplot
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch
from CPrinter import CPLevel
from FSelector import FSelector
from selection_methods.TE import TE, TEestimator
from selection_methods.MI import MI, MIestimator
from datetime import datetime


alpha = 0.05
min_lag = 1
max_lag = 1

resfolder = 'p12_TE_CMI'
np.random.seed(1)
nsample = 1500
nfeature = 12
data = np.random.randn(nsample, nfeature)


const = list()
for t in range(int(nsample/3)):
    const.append(5)
for t in range(int(nsample/3)):
    const.append(7)
for t in range(int(nsample/3)):
    const.append(2)
const = np.array(const)
data[:, -1] = const


for t in range(1, nsample):
    data[t, 0] += data[t-1, 1] * (4 * data[t-1, 4])
    data[t, 2] += 0.00003 * data[t-1, 1]**2
    data[t, 3] += data[t-1, 3] + 0.53
    data[t, 4] += 0.2 * (data[t-1, 4] + 2 * data[t-1, 5])
    data[t, 7] += 0.0002 * data[t-1, 6]
    data[t, 8] += 2 * data[t-1, 6]
    data[t, 9] += 0.75 * data[t-1, 6]
    data[t, 10] += data[t-1, 7] + data[t-1, 8]


var_names = ['X_' + str(f) for f in range(nfeature)]
var_names_pretty = [r'$X_' + str(f) + '$' for f in range(nfeature)]


df = pd.DataFrame(data, columns=var_names)
FS = FSelector(df, 
               alpha = alpha, 
               min_lag = min_lag, 
               max_lag = max_lag, 
               sel_method = TE(TEestimator.Kraskov), 
               verbosity = CPLevel.DEBUG,
               resfolder = resfolder)

startTE = datetime.now()
selector_res = FS.run()
stopTE = datetime.now()

print("\nTOTAL TIME WITH SELECTOR: (hh:mm:ss.ms) {}".format(str(stopTE - startTE)))

FS.print_dependencies()
FS.show_dependencies()


#___________________________________________________________________________________________

dataframe = pp.DataFrame(data, var_names = var_names_pretty)
# tigraplot.plot_timeseries(dataframe); plt.show()


# init and run pcmci
pcmci = PCMCI(dataframe = dataframe,
            #   cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
              cond_ind_test = CMIknn(),
              verbosity = 2)
              
startPCMCI = datetime.now()              
results = pcmci.run_pcmci(tau_max = 1,
                          tau_min = 1,
                          pc_alpha = 0.05)
stopPCMCI = datetime.now()              
print("\nTOTAL TIME WITHOUT SELECTOR: (hh:mm:ss.ms) {}".format(str(stopPCMCI - startPCMCI)))


tigraplot.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    save_name='results/'+resfolder+'/wo_selector',
    var_names=var_names_pretty,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    vmin_edges=0.,
    vmax_edges = 0.3,
    edge_ticks=0.05,
    cmap_edges='OrRd',
    vmin_nodes=0,
    vmax_nodes=.5,
    node_ticks=.1,
    cmap_nodes='OrRd',
    ); plt.show()