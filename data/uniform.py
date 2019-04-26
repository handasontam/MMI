import numpy as np
from scipy.special import xlogy

class Uniform():
    def __init__(self, mix, width_a, width_b, n_samples):
        self.mix, self.width_a, self.width_b, self.n_samples = mix, width_a, width_b, n_samples

    @property
    def data(self):
        mix, a, b = self.mix, self.width_a, self.width_b
        N1 = int(mix*self.n_samples)
        temp1 = np.array([[np.random.uniform(-a/2,a/2,N1)], [np.random.uniform(-.5/a,.5/a,N1)]]).T.reshape(N1,2)
        N2 = self.n_samples-N1
        temp2 = np.array([[np.random.uniform(-.5/b,.5/b,N2)], [np.random.uniform(-.5*b,.5*b,N2)]]).T.reshape(N2,2)
        #return np.append(temp1,temp2,axis = 0) 
        temp  = np.append(temp1,temp2,axis = 0) 
        np.random.shuffle(temp) 
        return temp

    @property
    def ground_truth(self):
        mix1, mix2, a, b = self.mix, 1-self.mix, self.width_a, self.width_b
        hX = -xlogy(mix2*(1-a*b),mix2*b) - (mix1+mix2*a*b)*np.log((mix1+mix2*a*b)/a)
        hY = -xlogy(mix1*(1-a*b),mix1*a) - (mix2+mix1*a*b)*np.log((mix2+mix1*a*b)/b)
#        temp1 = -(1-a*b)*(mix*np.log(a) + (1-mix)*np.log(b))
#        temp2 = -xlogy(mix+(1-mix)*a*b, mix/a+(1-mix)*b)
#        temp3 = -xlogy(mix*a*b+(1-mix), mix*a+(1-mix)/b)
#        mi = temp1+temp2+temp3
        hXY = -(1-a*b)*(xlogy(mix1,mix1) + xlogy(mix2,mix2))
        mi = hX+hY-hXY
        # return [mi, hXY, hX, hY]
        return mi


if __name__ == '__main__':
    data=Uniform(0.5, 2, 2, 200)
    x = data.data
    import matplotlib.pyplot as plt
    plt.scatter(x[:,0], x[:,1])
    print(data.ground_truth)
    plt.show()