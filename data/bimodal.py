import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy

class BiModal():
    # Mixture of two bivariate gaussians
    #
    # data(mix,Mode,Rho,N) generates N samples with
    # mix: mixing ration between 0 and 1 
    # Rho[0] correlation for the first bivariate gaussian and Rho[1] for the second
    # Mode[0] separation between the two bivariate gaussians along the x-axis and Mode[1] is the separation along the y-axis
    #
    # Adapated from Ali
    def __init__(self, n_samples=400, mean1=0, mean2=0, rho1=0.9, rho2=-0.9, mix=0.5, theta=0):
        self.n_samples = n_samples
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        covMat1 = np.array([[1,rho1],[rho1,1]])
        covMat2 = np.array([[1,rho2],[rho2,1]])
        self.n_samples = n_samples
        self.mix = mix
        self.covMat1 = np.matmul(np.matmul(R,covMat1), R.transpose())
        self.covMat2 = np.matmul(np.matmul(R,covMat2), R.transpose())
        self.mu = np.array([mean1, mean2])
        self.name = 'bimodal'

    @property
    def data(self):
        """[summary]
        
        Returns:
            [np.array] -- [N by 2 matrix]
        """
        N1 = int(self.mix*self.n_samples)
        N2 = self.n_samples-N1
        temp1 = np.random.multivariate_normal(mean=self.mu,
                                    cov=self.covMat1,
                                    size = N1)
        temp2 = np.random.multivariate_normal( mean=-self.mu,
                                    cov=self.covMat2,
                                    size = N2)
        X = np.append(temp1,temp2,axis = 0) 
        np.random.shuffle(X)
        return X

    @property
    def ground_truth(self):
        # MI(mix,Rho,Mode) computes mutual information
        # for mixture of two bivariate gaussians with
        # mix, Rho, and Mode as above.
        # Numerical integration may cause issues for Mode values above 10 and / or correlation above .99
        # (performance suffers for choices that results in large Mode[i]*Rho[i])
        # can possibly be resolved by avoiding exponentials or by using other integration methods
        mix, covMat1, covMat2, mu = self.mix, self.covMat1, self.covMat2, self.mu
        def fxy(x,y):
            X = np.array([x, y])
            temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat1)), (X-mu).transpose() )
            temp2 = np.matmul(np.matmul(X+mu , np.linalg.inv(covMat2)), (X+mu).transpose() )
            return mix*np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat1))) \
                    + (1-mix)*np.exp(-.5*temp2) / (2*np.pi * np.sqrt(np.linalg.det(covMat2)))

        def fx(x):
            return mix*np.exp(-(x-mu[0])**2/(2*covMat1[0,0])) / np.sqrt(2*np.pi*covMat1[0,0]) \
                    + (1-mix)*np.exp(-(x+mu[0])**2/(2*covMat2[0,0])) / np.sqrt(2*np.pi*covMat2[0,0])

        def fy(y):
            return mix*np.exp(-(y-mu[1])**2/(2*covMat1[1,1])) / np.sqrt(2*np.pi*covMat1[1,1]) \
                    + (1-mix)*np.exp(-(y+mu[1])**2/(2*covMat2[1,1])) / np.sqrt(2*np.pi*covMat2[1,1])


        lim = np.inf 
        hx = quad(lambda x: -xlogy(fx(x),fx(x)), -lim, lim)
        isReliable = hx[1]
        # print(isReliable)
        hy = quad(lambda y: -xlogy(fy(y),fy(y)), -lim, lim)
        isReliable = np.maximum(isReliable,hy[1])
        # print(isReliable)
        hxy = dblquad(lambda x, y: -xlogy(fxy(x,y),fxy(x,y)), -lim, lim, lambda x:-lim, lambda x:lim)
        isReliable = np.maximum(isReliable,hxy[1])
        return hx[0] + hy[0] - hxy[0]
        # return [hx[0] + hy[0] - hxy[0], isReliable, hx[0], hy[0]]

    def plot_i(self, ax, xs, ys):
        mix, covMat1, covMat2, mu = self.mix, self.covMat1, self.covMat2, self.mu
        def fxy(x,y):
            X = np.array([x, y])
            temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat1)), (X-mu).transpose() )
            temp2 = np.matmul(np.matmul(X+mu , np.linalg.inv(covMat2)), (X+mu).transpose() )
            return mix*np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat1))) \
                    + (1-mix)*np.exp(-.5*temp2) / (2*np.pi * np.sqrt(np.linalg.det(covMat2)))

        def fx(x):
            return mix*np.exp(-(x-mu[0])**2/(2*covMat1[0,0])) / np.sqrt(2*np.pi*covMat1[0,0]) \
                    + (1-mix)*np.exp(-(x+mu[0])**2/(2*covMat2[0,0])) / np.sqrt(2*np.pi*covMat2[0,0])

        def fy(y):
            return mix*np.exp(-(y-mu[1])**2/(2*covMat1[1,1])) / np.sqrt(2*np.pi*covMat1[1,1]) \
                    + (1-mix)*np.exp(-(y+mu[1])**2/(2*covMat2[1,1])) / np.sqrt(2*np.pi*covMat2[1,1])

        i = [np.log(fxy(xs[i,j], ys[i,j])/(fx(xs[i,j])*fy(ys[i,j]))) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        i = np.array(i).reshape(xs.shape[0], ys.shape[1])
        i = i[:-1, :-1]
        i_min, i_max = -np.abs(i).max(), np.abs(i).max()
        c = ax.pcolormesh(xs, ys, i, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c

if __name__ == '__main__':
    # data = BiModal().data
    # ground_truth = BiModal().ground_truth
    # from sklearn.feature_selection import mutual_info_regression
    # # print("data", data)
    # # print("ground truth", ground_truth)
    # import matplotlib.pyplot as plt
    # from sklearn import tree
    # import pylab as plt

    # plt.scatter(data[:,0], data[:,1])
    # plt.show()

    # N = 400
    # const = 2*np.pi*np.exp(1)
    # #correlations = np.linspace(1,1,1)

    # rho = .9
    # mix = .5
    # Mode = [0, 0]
    # Rho = [rho, -rho]
    # train_data = data
    # test_data = BiModal().data
    # x0_train = train_data[:,0].reshape(N,1)
    # x1_train = train_data[:,1].reshape(N,1)

    # x0 = test_data[:,0].reshape(N,1)
    # x1 = test_data[:,1].reshape(N,1)

    # fig, ax = plt.subplots()
    # plt.scatter(x0, x1, marker = 'o', facecolors='none', color = 'red') 
    # # x0 = x0_train
    # # x1 = x1_train
    # h_0 = .5*np.log(const*np.mean(np.square(x0 - x0.mean())) )
    # h_1 = .5*np.log(const*np.mean(np.square(x1 - x1.mean())) ) 

    # #mi_gaussian = -.5*np.log(1-np.square(rho))
    # #print('gaussian: ' + str(mi_gaussian))

    # ## true MI
    # mi_true= []
    # mi_true.append(ground_truth)
    # print('MI_true: ' + str(mi_true[-1]))
    # # print('true: h1 =  ' + str(tmi[3]) + ', h1_0 = ' + str(tmi[3]-tmi[0]) + ', is reliable: ' + str(tmi[1]))

    # ## decision tree 
    # mi_decTree = []

    # dtree0 = tree.DecisionTreeRegressor()
    # dtree1 = tree.DecisionTreeRegressor()
    # T0 = dtree0.fit(x1_train,x0_train)
    # T1 = dtree1.fit(x0_train,x1_train)

    # x0_Tpredict = T0.predict(x1).reshape(N,1) # predict x0 given x1
    # x1_Tpredict = T1.predict(x0).reshape(N,1) # predict x1 given x0


    # h_0_given_1 = .5*np.log(const*np.mean(np.square(x0 - x0_Tpredict)) )
    # h_1_given_0 = .5*np.log(const*np.mean(np.square(x1 - x1_Tpredict)) )

    # mi = h_1 - h_1_given_0
    # mi_decTree.append(mi)
    # print('MI_DecTree: ' + str(mi_decTree[-1]))
    # # print('DecTree: h1 =  ' + str(h_1) + ', h1_0 = ' + str(h_1_given_0))
    # plt.scatter(x0, x1_Tpredict, marker='+', color = 'blue')

    # ## linear model
    # mi_linReg = []
    # from sklearn.linear_model import LinearRegression
    # L1 = LinearRegression().fit(x0_train,x1_train)
    # x1_Lpredict = L1.predict(x0).reshape(N,1)
    # h_1_given_0 = .5*np.log(const*np.mean(np.square(x1 - x1_Lpredict)) )
    # mi = h_1 - h_1_given_0
    # mi_linReg.append(mi)
    # print('MI_LinReg: ' + str(mi_linReg[-1]))
    # # print('LinReg: h1 =  ' + str(h_1) + ', h1_0 = ' + str(h_1_given_0))
    # plt.scatter(x0, x1_Lpredict, marker='+', color = 'green')


    # print('MI_Kraskov:', mutual_info_regression(data[:, [0]], data[:, [1]]))
    # # ## NeurNet MINE
    # # import MINEbase
    # # # from MINEbase import *
    # # mi_MINE= []
    # # mine_net = MINEbase.Mine()
    # # mine_net_optim = MINEbase.optim.Adam(mine_net.parameters(), lr=1e-3)
    # # result = MINEbase.train(test_data,mine_net,mine_net_optim)
    # # result_ma = MINEbase.ma(result)
    # # mi_MINE.append(result_ma[-1])
    # # print('MI_NNet: ' + str(mi_MINE[-1]))

    


    # ax.legend(["data", "decTree", "linReg"], loc="upper right")
    # # ax.legend('aaa')
    # # ax.legend(loc='upper left')
    # ax.set_xlabel('x0')
    # ax.set_ylabel('x1')


    # plt.show(block=False)
    # input("press enter to continue")

    # for i in range(0, len(x0), 1):
    #     plt.plot([x0[i],x0[i]], [x1[i],x1_Tpredict[i]], 'b--')
    # # plt.plot([0,0],[1,0])

    # plt.show(block=False)
    # input("press enter to close")
    # plt.close(fig)
    bimodel=BiModal()
    data = bimodel.data
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
    ax, c = bimodel.plot_i(ax, xs, ys)
    fig.colorbar(c, ax=ax)
    ax.set_title("i(X;Y)")
    figName = os.path.join("experiments", "bimodel_rho=0.9_i_XY.png")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

    # plt.scatter(data[:,0], data[:,1])
    print(bimodel.ground_truth)
    # plt.show()