
from . import ifestimators as ife
import numpy as np
import math



class chiSqDiv:

    def __init__(self, numPart=1, numAvgPart=1, correctBound=False, Low=1e-5, Upp=math.inf, doAsympAnalysis=False, alpha=0.5):
        self.params = ife.Struct()
        self.funPara = ife.Struct()
        self.params['numPartitions'] = numPart
        self.params['numAvgPartitions'] = numAvgPart
        self.params['doBoundaryCorrection'] = correctBound
        self.params['estLowerBound'] = Low
        self.params['estUpperBound'] = Upp
        self.params['doAsympAnalysis'] = doAsympAnalysis
        self.params['alpha'] = alpha

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """

        return ife.octave.chiSqDivergence(X[:,0], X[:,1], self.funPara, self.params)
