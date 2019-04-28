import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy

class Uniform():
    def __init__(self, mix, width_a, width_b, n_samples):
        self.mix, self.width_a, self.width_b, self.n_samples = mix, width_a, width_b, n_samples

    @property
    def data(self):
        mix, a, b = self.mix, self.width_a, self.width_b
        N1 = int(mix*self.n_samples)
        temp1 = np.array([[np.random.uniform(-.5*a,.5*a,N1)], [np.random.uniform(-.5/a,.5/a,N1)]]).T.reshape(N1,2)
        N2 = self.n_samples-N1
        temp2 = np.array([[np.random.uniform(-.5/b,.5/b,N2)], [np.random.uniform(-.5*b,.5*b,N2)]]).T.reshape(N2,2)
        #return np.append(temp1,temp2,axis = 0) 
        temp  = np.append(temp1,temp2,axis = 0) 
        np.random.shuffle(temp) 
        return temp

    @property
    def ground_truth(self):
        def fx(x, mix_1, w_1_x, w_2_x):
            p = 0
            if abs(x) <= .5*w_1_x:
                p += mix_1/w_1_x
            if abs(x) <= .5*w_2_x:
                p += (1-mix_1)/w_2_x
            return p

        def fxy(x, y, mix_1, w_1_x, w_1_y, w_2_x, w_2_y):
            p = 0
            if abs(x) <= .5*w_1_x and abs(y) <= .5*w_1_y:
                p += mix_1/(w_1_x*w_1_y)
            if abs(x) <= .5*w_2_x and abs(y) <= .5*w_2_y:
                p += (1-mix_1)/(w_2_x*w_2_y)
            return p

        lim = np.inf 
        mix, a, b = self.mix, self.width_a, self.width_b
        hx = quad(lambda x: -xlogy(fx(x, mix, a, 1/b),fx(x, mix, a, 1/b)), -lim, lim)
        isReliable = hx[1]
        # print(isReliable)
        hy = quad(lambda y: -xlogy(fx(y, mix, 1/a, b),fx(y, mix, 1/a, b)), -lim, lim)
        isReliable = np.maximum(isReliable,hy[1])
        # print(isReliable)
        hxy = dblquad(lambda x, y: -xlogy(fxy(x,y, mix, a, 1/a, 1/b, b),fxy(x,y, mix, a, 1/a, 1/b, b)), -lim, lim, lambda x:-lim, lambda x:lim)
        isReliable = np.maximum(isReliable,hxy[1])
        mi = hx[0] + hy[0] - hxy[0]
        return mi

    def plot_i(self, ax, xs, ys):
        def fx(x, mix_1, w_1_x, w_2_x):
            p = 0
            if abs(x) <= .5*w_1_x:
                p += mix_1/w_1_x
            if abs(x) <= .5*w_2_x:
                p += (1-mix_1)/w_2_x
            return p

        def fxy(x, y, mix_1, w_1_x, w_1_y, w_2_x, w_2_y):
            p = 0
            if abs(x) <= .5*w_1_x and abs(y) <= .5*w_1_y:
                p += mix_1/(w_1_x*w_1_y)
            if abs(x) <= .5*w_2_x and abs(y) <= .5*w_2_y:
                p += (1-mix_1)/(w_2_x*w_2_y)
            return p

        mix, a, b = self.mix, self.width_a, self.width_b

        i = [np.log(fxy(xs[i,j], ys[i,j], mix, a, 1/a, 1/b, b)/(fx(xs[i,j], mix, a, 1/b)*fx(ys[i,j], mix, 1/a, b))) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        i = np.array(i).reshape(xs.shape[0], ys.shape[1])
        i = i[:-1, :-1]
        i_min, i_max = -np.abs(i).max(), np.abs(i).max()
        c = ax.pcolormesh(xs, ys, i, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c


if __name__ == '__main__':
    unif=Uniform(0.5, 10, 10, 200)
    data = unif.data
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
    ax, c = unif.plot_i(ax, xs, ys)
    fig.colorbar(c, ax=ax)
    ax.set_title("i(X;Y)")
    figName = os.path.join("", "i_XY")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

    # plt.scatter(data[:,0], data[:,1])
    print(unif.ground_truth)
    # plt.show()