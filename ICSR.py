from matplotlib import pyplot as plt
import numpy as np
import tigramite.data_processing as pp
import tigramite.plotting as tigraplot
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn, GPDC, GPDCtorch


alpha = 0.05
min_lag = 1
max_lag = 1

resfolder = 'ICSR'
np.random.seed(1)
nsample = 1500
nfeature = 5
data = np.random.randn(nsample, nfeature)


for t in range(1, nsample):
    data[t, 0] += data[t-1, 1] + (4 * data[t-1, 4])
    data[t, 1] += 0.5 * data[t-1, 1] + 0.3
    data[t, 2] += data[t-1, 0] + 0.2 * data[t-1, 3] + 0.053
    data[t, 3] += 0.2 * (data[t-1, 4] + 0.73 * data[t-1, 2])
    data[t, 4] += 0.8 * data[t-1, 3]


var_names = ['X_' + str(f) for f in range(nfeature)]
var_names_pretty = [r'$X_' + str(f) + '$' for f in range(nfeature)]

dataframe = pp.DataFrame(data, var_names = var_names_pretty)
tigraplot.plot_timeseries(dataframe); plt.show()

# init and run pcmci
pcmci = PCMCI(dataframe = dataframe,
              cond_ind_test = ParCorr(significance='analytic'),
            #   cond_ind_test = GPDC(significance = 'analytic', gp_params = None),
            #   cond_ind_test = CMIknn(),
              verbosity = 2)
                           
results = pcmci.run_pcmci(tau_max = max_lag,
                          tau_min = min_lag,
                          pc_alpha = alpha)


tigraplot.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    save_name='results/'+resfolder+'/dag',
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
    )


# save timeseries plot
tigraplot.plot_time_series_graph(figsize = (6, 4),
                                 graph = results['graph'],
                                 val_matrix = results['val_matrix'],
                                 var_names = var_names_pretty,
                                 link_colorbar_label = 'MCI',
                                 save_name = 'results/'+resfolder+'/ts_dag',
                                 edge_ticks = 0.2,
                                 vmin_edges = 0.,
                                 vmax_edges = round(results['val_matrix'].max(), 1),
                                 cmap_edges = 'OrRd',
                                 arrow_linewidth = 3); plt.show()