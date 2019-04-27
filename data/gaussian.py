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

    def plot_i(self, ax, xs, ys):
        covMat, mu = np.array([[1, self.rho], [self.rho, 1]]), np.array([self.mean1, self.mean2])
        def fxy(x,y):
            X = np.array([x, y])
            temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose() )
            return np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat))) 

        def fx(x):
            return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0]) 

        def fy(y):
            return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1]) 

        i = [np.log(fxy(xs[i,j], ys[i,j])/(fx(xs[i,j])*fy(ys[i,j]))) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        i = np.array(i).reshape(xs.shape[0], ys.shape[1])
        i = i[:-1, :-1]
        i_min, i_max = -np.abs(i).max(), np.abs(i).max()
        c = ax.pcolormesh(xs, ys, i, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c



if __name__ == '__main__':
    gaus=Gaussian(200, 0, 1, 0.5)
    data = gaus.data
    import os
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
        
    #Plot Ground Truth MI
    fig, axs = plt.subplots(1, 2, figsize=(45, 15))
    ax = axs[0]
    ax.scatter(data[:,0], data[:,1], color='r', marker='o')

    ax = axs[1]
    Xmax = max(data[:,0])+1
    Xmin = min(data[:,0])-1
    Ymax = max(data[:,1])+1
    Ymin = min(data[:,1])-1
    x = np.linspace(Xmin, Xmax, 300)
    y = np.linspace(Ymin, Ymax, 300)
    xs, ys = np.meshgrid(x,y)
    ax, c = gaus.plot_i(ax, xs, ys)
    fig.colorbar(c, ax=ax)
    ax.set_title("i(X;Y)")
    figName = os.path.join("experiments", "guas_rho=0.5_i_XY.png")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

    # plt.scatter(data[:,0], data[:,1])
    print(gaus.ground_truth)
    # plt.show()