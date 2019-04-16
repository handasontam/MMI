
from sklearn.tree import DecisionTreeRegressor
from .DiscreteCondEnt import computeEnt
from ..utils import mseEntropy, varEntropy, unifEntropy

class cartReg(DecisionTreeRegressor):

    def __init__(self, cvFold):
        super().__init__()
        self.cvFold = cvFold
        self.Reg = DecisionTreeRegressor()

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """

        cond_entropy = computeEnt(X, self.Reg, mseEntropy, varEntropy, self.cvFold)
        mutual_info = cond_entropy[1,0] + cond_entropy[0,0] - cond_entropy[0,1] - cond_entropy[1,1]
        mutual_info = mutual_info/2
        return mutual_info

    def predict_Cond_Entropy(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X n_variables]
        
        Returns:
            [numpy array] -- [n_variables X 2^n_variables]
        """

        cond_entropy = computeEnt(X, self.Reg, mseEntropy, unifEntropy, self.cvFold)
        return cond_entropy
