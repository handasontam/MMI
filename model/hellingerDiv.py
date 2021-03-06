from . import ifestimators as ife
import matlab
import numpy as np
import math



class hellingerDiv:

    def __init__(self, numPart=1, numAvgPart=1, correctBound=False, Low=1e-5, Upp=math.inf, doAsympAnalysis=False, alpha=0.5):
        # self.params = ife.Struct()
        # self.funPara = ife.Struct()
        self.params = dict()
        self.funPara = dict()
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

        # return ife.octave.hellingerDivergence(X[:,0], X[:,1], self.funPara, self.params)
        X0 = matlab.double(X[:,0][:,None].tolist())
        X1 = matlab.double(X[:,1][:,None].tolist())
        return ife.eng.hellingerDivergence(X0, X1, self.funPara, self.params)
