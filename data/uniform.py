import numpy as np
from scipy.special import xlogy

class Uniform():
    def __init__(self, mix, width_a, width_b, n_samples):
        self.mix, self.width_a, self.width_b, self.n_samples = mix, width_a, width_b, n_samples

    @property
    def data(self):
        mix, a, b = self.mix, self.width_a,self.width_b 
        N1 = int(mix*self.n_samples)
        temp1 = np.array([[np.random.uniform(-a/2,a/2,N1)], [np.random.uniform(-.5/a,.5/a,N1)]]).T.reshape(N1,2)
        N2 = self.n_samples-N1
        temp2 = np.array([[np.random.uniform(-.5/b,.5/b,N2)], [np.random.uniform(-.5*b,.5*b,N2)]]).T.reshape(N2,2)
        return np.append(temp1,temp2,axis = 0) 

    @property
    def ground_truth(self):
        mix, a, b = self.mix, self.width_a,self.width_b 
        temp1 = -(1-a*b)*(mix*np.log(a) + (1-mix)*np.log(b))
        temp2 = -xlogy(mix+(1-mix)*a*b, mix/a+(1-mix)*b)
        temp3 = -xlogy(mix*a*b+(1-mix), mix*a+(1-mix)/b)
        mi = temp1+temp2+temp3
        hXY = -(1-a*b)*(xlogy(mix,mix) + xlogy(1-mix,1-mix))
        # return [mi, hXY]
        return mi


if __name__ == '__main__':
    x=Uniform(0.5, 20, 2, 200).data
    import matplotlib.pyplot as plt
    plt.scatter(x[:,0], x[:,1])
    plt.show()