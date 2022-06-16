from selection_methods.SelectionMethod import SelectionMethod
import selection_methods.constants as const
from sklearn.feature_selection import SelectKBest, f_regression


class CCorr(SelectionMethod):
    def __init__(self):
        super().__init__(const.score_fcn.CCorr)


    def compute_dependencies(self):
        # TODO: add lag dependencies
        for target in self.features:
            X, Y = self.prepare_ts(target)

            selector = SelectKBest(f_regression, k = const.K_BEST)
            selector.fit_transform(X, Y)

            # TODO: decide if it is to remove
            if const.K_BEST != 'all':
                sel_sources = X.columns[selector.get_support(indices=True)].tolist()
                sel_sources_score = selector.scores_.tolist()
                sel_sources_pval = selector.pvalues_.tolist()
            else:
                # Filter on pvalue
                f = selector.pvalues_ < self.alpha

                # Result of the selection
                sel_sources = X.columns[f].tolist()
                sel_sources_score = selector.scores_[f].tolist()
                sel_sources_pval = selector.pvalues_[f].tolist()

            for s, score, pval in zip(sel_sources, sel_sources_score, sel_sources_pval):
                self.add_dependecies(target, s, score, pval, self.lag)

        return self.result