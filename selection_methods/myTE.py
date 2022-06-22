import numpy as np
from selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
import jpype

from selection_methods.constants import *
import copy


# Change location of jar to match yours:
jarLocation = "jidt/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)



class myTE(SelectionMethod):
    def __init__(self):
        super().__init__(CTest.TE)
        seed = 42
        self.random_state = np.random.default_rng(seed)
        self.EstimTE = self.estim_init()


    def estim_init(self):
        """
        Initialize JIDT estimator

        Returns:
            _type_: JIDT TE estimator
        """
        # Create a Kraskov TE calculator:
        teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
        teCalc = teCalcClass()

        # Normalise the individual variables
        teCalc.setProperty("NORMALISE", "true") 

        # Property name to specify the destination history embedding length k
        teCalc.setProperty(teCalcClass.K_PROP_NAME, "1")

        # Property name for embedding delay for the destination past history vector
        teCalc.setProperty(teCalcClass.K_TAU_PROP_NAME, "1")

        # Property name for embedding length for the source past history vector
        teCalc.setProperty(teCalcClass.L_PROP_NAME, "1")

        # Property name for embedding delay for the source past history vector
        teCalc.setProperty(teCalcClass.L_TAU_PROP_NAME, "1")

        # Property name for source-destination delay
        teCalc.setProperty(teCalcClass.DELAY_PROP_NAME, "1")
        
        teCalc.initialise()
        return teCalc


    def measure_dependecy(self, source, target):
        """
        Measures Transfer entropy from source to target

        Args:
            source (array): source timeseries
            target (array): target timeseries

        Returns:
            float: transfer entropy from source to target
        """
        self.EstimTE.setObservations(source, target)
        return self.EstimTE.computeAverageLocalOfObservations()


    def measure_significance(self, significant):
        """
        Measure significance of the measure previously performed

        Args:
            significant (float): dependency value

        Returns:
            float: p-value associated to the dependency value
        """
        null_dist = np.zeros(1000)
        for i in range(len(null_dist)):
            surrogates = self.random_state.random((self.n_samples, 2))
            null_dist[i] = self.measure_dependecy(surrogates[:,0], surrogates[:,1])
        return sum(null_dist > significant)/len(null_dist)


    def measure_autocorrelation(self, t):
        """
        Transfer entropy to measure autocorrelation

        Args:
            t (array): target timeseries

        Returns:
            (float, float): Transfer entropy and p-value
        """
        te = self.measure_dependecy(t, t)
        pval = self.measure_significance(te)
        return te, pval


    def compute_dependencies(self):
        sources = {f:list() for f in self.features}
        for t in self.features:
            te, pval = self.measure_autocorrelation(self.d[t].values)
            if pval < self.alpha:
                sources[t].append({SOURCE:t, 
                                   SCORE:te,
                                   PVAL:pval, 
                                   LAG:1})

            candidates = copy.deepcopy(self.features)
            candidates.remove(t)

            for c in candidates:
                conditions_vars = [s[SOURCE] for s in sources[t]]
                if len(conditions_vars) != 0:
                    source = np.c_[self.d[conditions_vars], self.d[c].values]
                    self.EstimTE.initialise(source.shape[1], source.shape[1])
                    self.EstimTE.addObservations(source, self.d[t].values)
                    result = self.EstimTE.computeAverageLocalOfObservations()
                else:
                    source = self.d[c].values
                te = self.measure_dependecy(source, self.d[t].values)
                pval = self.measure_significance(te)
                if pval < self.alpha:
                    sources[t].append({SOURCE:c, 
                                       SCORE:te,
                                       PVAL:pval, 
                                       LAG:1})
            

                # self._add_dependecies(t, s, te, pval, self.max_lag)


        return self.result