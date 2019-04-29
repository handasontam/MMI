import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from .pytorchtools import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
# from .DiscreteCondEnt import subset
import os
from ..util import plot_util
from ..util import torch_util

# from ..utils import save_train_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def sample_batch(data, resp=0, cond=[1], batch_size=100, sample_mode='marginal'):
    """[summary]
    
    Arguments:
        data {[type]} -- [N X 2]
        resp {[int]} -- [description]
        cond {[list]} -- [1 dimension]
    
    Keyword Arguments:
        batch_size {int} -- [description] (default: {100})
        randomJointIdx {bool} -- [description] (default: {True})
    
    Returns:
        [batch_joint] -- [batch size X 2]
        [batch_mar] -- [batch size X 2]
    """
    if type(cond)==list:
        whole = cond.copy()
        whole.append(resp)
    else:
        raise TypeError("cond should be list")
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
        batch = batch[:, whole]
    elif sample_mode == 'unif':
        dataMax = data.max(axis=0)[whole]
        dataMin = data.min(axis=0)[whole]
        batch = (dataMax - dataMin)*np.random.random((batch_size,len(cond)+1)) + dataMin
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,resp].reshape(-1,1), data[marginal_index][:,cond].reshape(-1,len(cond))], axis=1)
    else:
        raise ValueError('Sample mode: {} not recognized.'.format(sample_mode))
    return batch


class MineMultiTaskNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc_x = nn.Linear(1, hidden_size)
        self.fc_x_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_x_3 = nn.Linear(hidden_size, 1)

        self.fc_y = nn.Linear(input_size, hidden_size)
        self.fc_y_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_y_3 = nn.Linear(hidden_size, 1)

        self.fc_xy = nn.Linear(input_size, hidden_size)
        self.fc_xy_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_xy_3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        x = F.elu(self.fc_x(input[:,0]))
        x_hidden = F.elu(self.fc_x_2(x))
        x_output = F.elu(self.fc_x_3(x_hidden))

        y = F.elu(self.fc_y(input[:,1]))
        y_hidden = F.elu(self.fc_y_2(y))
        y_output = F.elu(self.fc_y_3(y_hidden))

        xy = F.elu(self.fc_xy(input)) + x_hidden + y_hidden
        xy_hidden = F.elu(self.fc_xy_2(xy))
        xy_output = F.elu(self.fc_xy_3(xy_hidden))

        return x_output, y_output, xy_output

class MineMultiTask():
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, verbose=True, resp=0, cond=[1], log=True, sample_mode='marginal', y_label=""):
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience  # 20
        self.iter_num = iter_num  # 1e+3
        self.log_freq = int(log_freq)  # int(1e+2)
        self.avg_freq = avg_freq  # int(1e+1)
        self.ma_rate = ma_rate  # 0.01
        self.prefix = ''
        self.verbose = verbose
        self.resp = resp
        self.cond = cond
        self.log = log
        self.sample_mode = sample_mode 
        self.model_name = ""
        self.ground_truth = None
        self.paramName = None
        if sample_mode == "marginal":
            self.y_label = "I(X^Y)"
        elif sample_mode == "unif":
            self.y_label = "HXY"
        else:
            self.y_label = y_label
        self.heatmap_frames = []  # for plotting heatmap animation
        self.mine_net = MineMultiTaskNet(input_size=len(self.cond)+1)
        self.mine_net_optim = optim.Adam(self.mine_net.parameters(), lr=self.lr)

    def fit(self, train_data, val_data):
        self.Xmin = min(train_data[:,0])
        self.Xmax = max(train_data[:,0])
        self.Ymin = min(train_data[:,1])
        self.Ymax = max(train_data[:,1])
    
        if self.log:
            log_file = os.path.join(self.prefix, "MINE_train.log")
            log = open(log_file, "w")
            log.write("batch_size={0}\n".format(self.batch_size))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("log_freq={0}\n".format(self.log_freq))
            log.write("avg_freq={0}\n".format(self.avg_freq))
            log.write("patience={0}\n".format(self.patience))
            log.close()
            heatmap_animation_fig, heatmap_animation_ax = plt.subplots(1, 1)
        # data is x or y
        result = list()
        self.ma_et = 1.  # exponential of mi estimation on marginal data
        
        #Early Stopping
        train_mi_lb = []
        valid_mi_lb = []
        self.avg_train_mi_lb = []
        self.avg_valid_mi_lb = []
        
        earlyStop = EarlyStopping(patience=self.patience, verbose=self.verbose, prefix=self.prefix)
        for i in range(self.iter_num):
            #get train data
            batchTrain = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode='joint'), \
                         sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode=self.sample_mode)
            mi_lb, lossTrain = self.update_mine_net(batchTrain, self.mine_net_optim)
            result.append(mi_lb.detach().cpu().numpy())
            train_mi_lb.append(mi_lb.item())
            if self.verbose and (i+1)%(self.log_freq)==0:
                print(result[-1])
            
            mi_lb_valid = self.forward_pass(val_data)
            valid_mi_lb.append(mi_lb_valid.item())
            
            if (i+1)%(self.avg_freq)==0:
                train_loss = - np.average(train_mi_lb)
                valid_loss = - np.average(valid_mi_lb)
                self.avg_train_mi_lb.append(train_loss)
                self.avg_valid_mi_lb.append(valid_loss)

                if self.verbose:
                    print_msg = "[{0}/{1}] train_loss: {2} valid_loss: {3}".format(i, self.iter_num, train_loss, valid_loss)
                    print (print_msg)

                train_mi_lb = []
                valid_mi_lb = []

                earlyStop(valid_loss, self.mine_net)
                if (earlyStop.early_stop):
                    if self.verbose:
                        print("Early stopping")
                    break
                if self.log:
                    x = np.linspace(self.Xmin, self.Xmax, 300)
                    y = np.linspace(self.Ymin, self.Ymax, 300)
                    xs, ys = np.meshgrid(x,y)
                    t = self.mine_net(torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
                    # ixy = t - np.log(self.ma_et.mean().detach().numpy())
                    heatmap_animation_ax, c = plot_util.getHeatMap(heatmap_animation_ax, xs, ys, t)
                    self.heatmap_frames.append((c,))
    
        if self.log:
            writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
            heatmap_animation = animation.ArtistAnimation(heatmap_animation_fig, self.heatmap_frames, interval=200, blit=False)
            heatmap_animation.save(os.path.join(self.prefix, 'heatmap.mp4'), writer=writer)
            #Save result to files
            avg_train_mi_lb = np.array(self.avg_train_mi_lb)
            np.savetxt(os.path.join(self.prefix, "avg_train_mi_lb.txt"), avg_train_mi_lb)
            avg_valid_mi_lb = np.array(self.avg_valid_mi_lb)
            np.savetxt(os.path.join(self.prefix, "avg_valid_mi_lb.txt"), avg_valid_mi_lb)

        ch = os.path.join(self.prefix, "checkpoint.pt")
        self.mine_net.load_state_dict(torch.load(ch))#'checkpoint.pt'))

    
    def update_mine_net(self, batch, mine_net_optim, ma_rate=0.01):
        """[summary]
        
        Arguments:
            batch {[type]} -- ([batch_size X 2], [batch_size X 2])
            mine_net_optim {[type]} -- [description]
            ma_rate {float} -- [moving average rate] (default: {0.01})
        
        Keyword Arguments:
            mi_lb {} -- []
        """

        # batch is a tuple of (joint, marginal)
        joint , reference = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        reference = torch.autograd.Variable(torch.FloatTensor(reference))
        mi_lb, fx, fy, fxy, efx, efy, efxy = self.mutual_information(joint, reference)

        self.ma_efx = (1-ma_rate)*self.ma_efx + ma_rate*torch.mean(efx)
        self.ma_efy = (1-ma_rate)*self.ma_efy + ma_rate*torch.mean(efy)
        self.ma_efxy = (1-ma_rate)*self.ma_efxy + ma_rate*torch.mean(efxy)
        
        # unbiasing use moving average
        loss = -(torch.mean(fx) - (1/self.ma_efx.mean()).detach()*torch.mean(efx)) \
               -(torch.mean(fy) - (1/self.ma_efy.mean()).detach()*torch.mean(efy)) \
               -(torch.mean(fxy) - (1/self.ma_efxy.mean()).detach()*torch.mean(efxy))

        lossTrain = loss
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, lossTrain

    def mutual_information(self, joint, reference):
        fx, fy, fxy = self.mine_net(joint)
        efx, efy, efxy = torch.exp(self.mine_net(reference))
        h_x = torch.mean(fx) - torch.log(torch.mean(efx))
        h_y = torch.mean(fy) - torch.log(torch.mean(efy))
        h_xy = torch.mean(fxy) - torch.log(torch.mean(efxy))
        mi_lb = h_x + h_y - h_xy
        return mi_lb, fx, fy, fxy, efx, efy, efxy

    def forward_pass(self, X):
        joint = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], sample_mode='joint')
        reference = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], sample_mode=self.sample_mode)
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        reference = torch.autograd.Variable(torch.FloatTensor(reference))
        mi_lb, fx, fy, fxy, efx, efy, efxy = self.mutual_information(joint, reference)
        return mi_lb

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """
        self.X = X
        X_train, X_test = train_test_split(X, test_size=0.35, random_state=0)
        self.fit(X_train, X_test)
    
        mi_lb = self.forward_pass(X_test).item()

        if self.log:
            self.savefig(X, mi_lb)
        if self.sample_mode == 'unif':
            if 0 == len(self.cond):
                X_max, X_min = X[:,self.resp].max(axis=0), X[:,self.resp].min(axis=0)
                cross = np.log(X_max-X_min)
            else:
                X_max, X_min = X.max(axis=0), X.min(axis=0)
                cross = sum(np.log(X_max-X_min))
            return cross - mi_lb
        return mi_lb


    def savefig(self, X, ml_lb_estimate):
        if len(self.cond) > 1:
            raise ValueError("Only support 2-dim or 1-dim")
        fig, ax = plt.subplots(1,4, figsize=(90, 15))
        #plot Data
        ax[0].scatter(X[:,self.resp], X[:,self.cond], color='red', marker='o')

        #plot training curve
        ax[1] = plot_util.getTrainCurve(self.avg_train_mi_lb, self.avg_valid_mi_lb, ax[1])

        # Trained Function contour plot
        Xmin = min(X[:,0])
        Xmax = max(X[:,0])
        Ymin = min(X[:,1])
        Ymax = max(X[:,1])
        x = np.linspace(Xmin, Xmax, 300)
        y = np.linspace(Ymin, Ymax, 300)
        xs, ys = np.meshgrid(x,y)
        z = self.mine_net(torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
        ax[2], c = plot_util.getHeatMap(ax[2], xs, ys, z)

        fig.colorbar(c, ax=ax[2])
        ax[2].set_title('heatmap')

        # Plot result with ground truth
        ax[3].scatter(0, self.ground_truth, edgecolors='red', facecolors='none', label='Ground Truth')
        ax[3].scatter(0, ml_lb_estimate, edgecolors='green', facecolors='none', label=self.model_name)
        ax[3].set_xlabel(self.paramName)
        ax[3].set_ylabel(self.y_label)
        ax[3].legend()
        figName = os.path.join(self.prefix, "MINE")
        fig.savefig(figName, bbox_inches='tight')
        plt.close()
        





