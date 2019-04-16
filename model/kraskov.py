
from sklearn.feature_selection import mutual_info_regression
import numpy as np

class Kraskov():
    def __init__(self, discrete_features, n_neighbors, random_state=None):
        """
        discrete_features : {‘auto’, bool, array_like}, default ‘auto’
        If bool, then determines whether to consider all features discrete or continuous. If array, then it should be either a boolean mask with shape (n_features,) or array with indices of discrete features. If ‘auto’, it is assigned to False for dense X and to True for sparse X.

        n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables, see [2] and [3]. Higher values reduce variance of the estimation, but could introduce a bias.

        random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator for adding small noise to continuous variables in order to remove repeated values. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
        """
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[np array]} -- [N X 2]

        Returns:
            mutual information estimation
        """
        
        return mutual_info_regression(X=X[:, [0]], 
                                      y=X[:, 1], 
                                      discrete_features=self.discrete_features, 
                                      n_neighbors=self.n_neighbors, 
                                      random_state=self.random_state)[0]