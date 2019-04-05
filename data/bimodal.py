import numpy as np

class BiModal():
    def __init__(self, n_samples, mean, covariance):
        self.n_samples = n_samples
        self.mean = mean
        self.covariance = covariance

    @property
    def data(self):
        """[summary]
        
        Returns:
            [np.array] -- [N by 2 matrix]
        """

        x1 = np.random.multivariate_normal( mean=self.mean, cov=[[1,self.covariance],[self.covariance,1]], size=self.n_samples//2)
        x2 = np.random.multivariate_normal( mean=self.mean, cov=[[1,-1*self.covariance],[-1*self.covariance,1]], size=self.n_samples//2)
        return np.vstack((x1, x2))
    
    @property
    def ground_truth(self):
        raise NotImplementedError