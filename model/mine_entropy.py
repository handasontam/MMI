import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from .mine import Mine, MineNet, sample_batch

class Mine_ent(Mine):
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, verbose=True, resp=0, cond=[1], log=True):
        """
        can only support bivariate mutual information now
        """
        self.log_ent = log
        super().__init__(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, verbose, resp, cond, log=False, sample_mode='unif')

        self.Mine_resp = Mine(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, verbose, resp, [], log=False, sample_mode='unif')

        self.Mine_cond = Mine(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, verbose, cond[0], [], log=False, sample_mode='unif')

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """
        self.Mine_resp.prefix = os.path.join(self.prefix, 'X')
        os.makedirs(self.Mine_resp.prefix)
        HX = self.Mine_resp.predict(X)

        self.Mine_cond.prefix = os.path.join(self.prefix, 'Y')
        os.makedirs(self.Mine_cond.prefix)
        HY = self.Mine_cond.predict(X)

        self.prefix = os.path.join(self.prefix, 'XY')
        os.makedirs(self.prefix)
        HXY = super(Mine_ent, self).predict(X)
        mi_lb = HX + HY - HXY
        if self.log_ent:
            self.savefig(X, mi_lb)
        return mi_lb


    def savefig(self, X, ml_lb_estimate):
        if len(self.cond) > 1:
            raise ValueError("Only support 2-dim or 1-dim")
        fig, ax = plt.subplots(3,4, figsize=(100, 45))
        # fig, ax = plt.subplots(3,3, figsize=(60, 90))
        #plot Data
        axCur = ax[0,0]
        axCur.scatter(X[:,self.resp], X[:,self.cond], color='red', marker='o')
        axCur.set_title('scatter plot of data')

        #plot training curve
        axCur = ax[0,1]
        axCur = super(Mine_ent, self).getTrainCurve(axCur)
        axCur.set_title('train curve of HXY')
        axCur = ax[1,1]
        axCur = self.Mine_resp.getTrainCurve(axCur)
        axCur.set_title('train curve of HX')
        axCur = ax[2,1]
        axCur = self.Mine_cond.getTrainCurve(axCur)
        axCur.set_title('train curve of HXY')

        # Trained Function contour plot
        Xmin = min(X[:,0])
        Xmax = max(X[:,0])
        Ymin = min(X[:,1])
        Ymax = max(X[:,1])
        x = np.linspace(Xmin, Xmax, 300)
        y = np.linspace(Ymin, Ymax, 300)
        xs, ys = np.meshgrid(x,y)

        axCur = ax[0,2]
        axCur, HXY, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, sampleNum=5)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap T(X,Y)')

        axCur = ax[0,3]
        axCur, _, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, Z=HXY)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap H(X,Y)')

        axCur = ax[1,2]
        axCur, HX = self.Mine_resp.getResultPlot(axCur, x, sampleNum=5)
        axCur.set_title('plot of T(X)')

        axCur = ax[1,3]
        axCur, _ = self.Mine_resp.getResultPlot(axCur, x, Z=HX)
        axCur.set_title('plot of H(X)')

        axCur = ax[2,2]
        axCur, HY = self.Mine_cond.getResultPlot(axCur, y, sampleNum=5)
        axCur.set_title('plot of T(Y)')

        axCur = ax[2,3]
        axCur, _ = self.Mine_resp.getResultPlot(axCur, y, Z=HY)
        axCur.set_title('plot of H(Y)')

        axCur = ax[1,0]
        HX = HX[:-1]
        HY = HY[:-1]
        MI_XY = [HX[i]+HY[j]-HXY[i,j] for i in range(HX.shape[0]) for j in range(HY.shape[0])]
        MI_XY = np.array(MI_XY).reshape(HX.shape[0], HY.shape[0])
        axCur, _, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, MI_XY)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap of MI_XY')


        # Plot result with ground truth
        axCur = ax[2,0]
        axCur.scatter(0, self.ground_truth, edgecolors='red', facecolors='none', label='Ground Truth')
        axCur.scatter(0, ml_lb_estimate, edgecolors='green', facecolors='none', label="MINE_{0}".format(self.model_name))
        axCur.set_xlabel(self.paramName)
        axCur.set_ylabel(self.y_label)
        axCur.legend()
        axCur.set_title('MI of XY')
        figName = os.path.join(self.prefix, "MINE")
        fig.savefig(figName, bbox_inches='tight')
        plt.close()
