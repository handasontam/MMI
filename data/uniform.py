import numpy as np
from numpy.random import uniform as unif

class Uniform():
    def __init__(self, n_samples, n_variables=3, low=0.0, high=1.0, varName="", varValue=0):
        self.n_variables = int(n_variables)
        self.n_samples = int(n_samples)
        self.low = low
        self.high = high
        self.varName = varName
        self.varValue = varValue
        self.name = 'uniform'

    @property
    def data(self):
        """[summary]
        
        Raises:
            ValueError -- [if number of variable is less than 3, insuffieint to test MMI]
        
        Returns:
            [np array] -- [n_sample by n_variables matrix]
        """

        if self.n_variables < 2 or self.n_samples < 1:
            raise ValueError
        else:
            x = unif(self.low, self.high, self.n_samples*(self.n_variables-1)).reshape(self.n_samples, self.n_variables-1)
            x = np.append(x, np.remainder(np.sum(x, axis=1),1)[:,None],axis=1)
            return x
            
    @property
    def ground_truth(self):
        return np.log(self.high-self.low)

