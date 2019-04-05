import numpy as np

class Gaussian():
    def __init__(self, n_samples, mean, covariance):
        self.n_samples = n_samples
        self.mean = mean
        self.covariance = covariance

    @property
    def data(self):
        """[summary]
        
        Returns:
            [np array] -- [N by 2 matrix]
        """

        return np.random.multivariate_normal(mean=self.mean, cov=[[1, self.covariance], [self.covariance, 1]], size = self.n_samples)

    @property
    def ground_truth(self):
        return -0.5*np.log(1-self.covariance*self.covariance)

