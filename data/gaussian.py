import numpy as np

class Gaussian():
    def __init__(self, n_samples, mean1, mean2, rho, varValue=0):
        self.n_samples = n_samples
        self.mean1 = mean1
        self.mean2 = mean2
        self.rho = rho

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """

        return np.random.multivariate_normal(
            mean=[self.mean1, self.mean2],
            cov=[[1, self.rho], [self.rho, 1]], 
            size=self.n_samples)

    @property
    def ground_truth(self):
        return -0.5*np.log(1-self.rho*self.rho)

