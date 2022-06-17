from selection_methods.SelectionMethod import SelectionMethod, CTest
from sklearn.feature_selection import f_regression


class CCorr(SelectionMethod):
    def __init__(self):
        super().__init__(CTest.CCorr)


    def compute_dependencies(self):
        for lag in range(self.min_lag, self.max_lag + 1):
            for target in self.features:
                X, Y = self._prepare_ts(target, lag)

                scores, pval = f_regression(X, Y)

                # Filter on pvalue
                f = pval < self.alpha

                # Result of the selection
                sel_sources = X.columns[f].tolist()
                sel_sources_score = scores[f].tolist()
                sel_sources_pval = pval[f].tolist()

                for s, score, pval in zip(sel_sources, sel_sources_score, sel_sources_pval):
                    self._add_dependecies(target, s, score, pval, lag)

        return self.result