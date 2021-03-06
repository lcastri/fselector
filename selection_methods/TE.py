from selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data


class TE(SelectionMethod):
    def __init__(self):
        super().__init__(CTest.TE)


    def compute_dependencies(self):
        with _suppress_stdout():
            data = Data(self.d.values, dim_order='sp') # sp = samples(row) x processes(col)

            network_analysis = MultivariateTE()
            settings = {'cmi_estimator': 'JidtGaussianCMI',
                        'max_lag_sources': self.max_lag,
                        'min_lag_sources': self.min_lag,
                        'alpha_max_stats': self.alpha,
                        'alpha_min_stats': self.alpha,
                        'alpha_omnibus': self.alpha,
                        'alpha_max_seq': self.alpha,
                        'permute_in_time': False,
                        'verbose': False}
            results = network_analysis.analyse_network(settings=settings, data=data)
            
        for t in results._single_target.keys():
            sel_sources = [s[0] for s in results._single_target[t]['selected_vars_sources']]
            if sel_sources:
                sel_sources_lag = [s[1] for s in results._single_target[t]['selected_vars_sources']]
                sel_sources_score = results._single_target[t]['selected_sources_te']
                sel_sources_pval = results._single_target[t]['selected_sources_pval']
                for s, score, pval, lag in zip(sel_sources, sel_sources_score, sel_sources_pval, sel_sources_lag):
                    self._add_dependecies(self.features[t], self.features[s], score, pval, lag)

        return self.result