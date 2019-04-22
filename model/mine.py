import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from .pytorchtools import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from ..MMI.IC.AIC import TableEntropy
# from .DiscreteCondEnt import subset
import os

from ..utils import save_train_curve
# import matplotlib.pyplot as plt


# def save_train_curve(train_loss, valid_loss, figName):
#     # visualize the loss as the network trained
#     fig = plt.figure(figsize=(10,8))
#     plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
#     plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

#     # find position of lowest validation loss
#     minposs = valid_loss.index(min(valid_loss))+1 
#     plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.xlim(0, len(train_loss)+1) # consistent scale
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     fig.savefig(figName, bbox_inches='tight')
#     plt.close()


def sample_joint_marginal(data, resp=0, cond=[1], batch_size=100, marginal_mode='shuffle'):
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
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch_joint = data[index]
    if type(cond)==list:
        whole = cond.copy()
        whole.append(resp)
        batch_joint = batch_joint[:, whole]
    else:
        raise TypeError("cond should be list")
    if 'unif' == marginal_mode:
        dataMax = data.max(axis=0)[whole]
        dataMin = data.min(axis=0)[whole]
        batch_mar = (dataMax - dataMin)*np.random.random((batch_size,len(cond)+1)) + dataMin
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch_mar = np.concatenate([data[joint_index][:,resp].reshape(-1,1), data[marginal_index][:,cond].reshape(-1,len(cond))], axis=1)
    return batch_joint, batch_mar


class MineNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class Mine():
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, prefix="", verbose=True, resp=0, cond=[1], log=True, marginal_mode='shuffle'):
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience  # 20
        self.iter_num = iter_num  # 1e+3
        self.log_freq = int(log_freq)  # int(1e+2)
        self.avg_freq = avg_freq  # int(1e+1)
        self.ma_rate = ma_rate  # 0.01
        self.prefix = prefix
        self.verbose = verbose
        self.resp = resp
        self.cond = cond
        self.log = log
        self.marginal_mode = marginal_mode

    def fit(self, train_data, val_data):
        self.mine_net = MineNet(input_size=len(self.cond)+1)
        self.mine_net_optim = optim.Adam(self.mine_net.parameters(), lr=self.lr)
    
        if self.log:
            logName = "{0}MINE_train.log".format(self.prefix)
            log = open(logName, "w")
            log.write("batch_size={0}\n".format(self.batch_size))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("log_freq={0}\n".format(self.log_freq))
            log.write("avg_freq={0}\n".format(self.avg_freq))
            log.write("patience={0}\n".format(self.patience))
            log.close()
        # data is x or y
        result = list()
        ma_et = 1.
        
        #Early Stopping
        train_mi_lb = []
        valid_mi_lb = []
        self.avg_train_mi_lb = []
        self.avg_valid_mi_lb = []
        
        earlyStop = EarlyStopping(patience=self.patience, verbose=self.verbose, prefix=self.prefix)
        for i in range(self.iter_num):
            #get train data
            batchTrain = sample_joint_marginal(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, marginal_mode=self.marginal_mode)
            mi_lb, ma_et, lossTrain = self.update_mine_net(batchTrain, self.mine_net_optim, ma_et)
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
            #Save result to files
            avg_train_mi_lb = np.array(self.avg_train_mi_lb)
            np.savetxt("{0}avg_train_mi_lb.txt".format(self.prefix), avg_train_mi_lb)
            avg_valid_mi_lb = np.array(self.avg_valid_mi_lb)
            np.savetxt("{0}avg_valid_mi_lb.txt".format(self.prefix), avg_valid_mi_lb)

        ch = "{0}checkpoint.pt".format(self.prefix)
        self.mine_net.load_state_dict(torch.load(ch))#'checkpoint.pt'))

    
    def update_mine_net(self, batch, mine_net_optim, ma_et, ma_rate=0.01):
        """[summary]
        
        Arguments:
            batch {[type]} -- ([batch_size X 2], [batch_size X 2])
            mine_net_optim {[type]} -- [description]
            ma_et {[float]} -- [exponential of mi estimation on marginal data]
            ma_rate {float} -- [moving average rate] (default: {0.01})
        
        Keyword Arguments:
            ma_et {float} -- [exponential of mi estimation on marginal data]
            mi_lb {} -- []
        """

        # batch is a tuple of (joint, marginal)
        joint , marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        mi_lb , t, et = self.mutual_information(joint, marginal, self.mine_net)
        ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
        
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
    #     loss = - mi_lb
        lossTrain = loss
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et, lossTrain
    
    def mutual_information(self, joint, marginal, mine_net):
        t = self.mine_net(joint)
        et = torch.exp(self.mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def forward_pass(self, X):
        joint , marginal = sample_joint_marginal(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], marginal_mode=self.marginal_mode)
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        mi_lb , t, et = self.mutual_information(joint, marginal, self.mine_net)
        return mi_lb

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """
        X_train, X_test = train_test_split(
            X, test_size=0.35, random_state=0)
        self.fit(X_train, X_test)
    
        mi_lb = self.forward_pass(X_test)
        return mi_lb.item()
        
        # joint , marginal = sample_joint_marginal(X, batch_size=X.shape[0], resp=self.resp, cond=self.cond)
        # joint = torch.autograd.Variable(torch.FloatTensor(joint))
        # marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        # mi_lb , t, et = self.mutual_information(joint, marginal, self.mine_net)
        # return -mi_lb


    def predict_Cond_Entropy(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            cond_Ent_tablem {[np array]} -- [numVar X numCondset]
        """
        X_train, X_test = train_test_split(
            X, test_size=0.35, random_state=0)
        n_var = X.shape[1]
        numCond = 2**(n_var-1)
        cond_ent_mine = np.zeros((n_var, numCond))
        prefix_base = self.prefix
        prefix_name = "{0}n_var={1}/mine".format(prefix_base, n_var)
        for Resp in range(n_var):
            for sI in range(1, numCond):
                subset = TableEntropy.subsetVector(n_var - 1, sI)
                # subset = subset(n_var - 1, sI)
                subset = np.array(subset)
                cond = []
                for element in subset:
                    if element >= Resp:
                        element += 1
                    cond.append(int(element))
                prefix_name_loop = "{0}_resp={1}_cond={2}/".format(prefix_name, Resp, cond)
                os.mkdir(prefix_name_loop)
                self.prefix = prefix_name_loop
                self.resp = Resp
                self.cond = cond
                self.fit(X_train, X_test)
                self.savefig()
                cond_ent_mine[Resp, sI] = self.forward_pass(X_test).item() + cond_ent_mine[Resp, 0]

        self.prefix = prefix_base
        return cond_ent_mine


    def savefig(self):
        figName = "{0}trainLog_resp={1}_cond={2}.png".format(self.prefix, self.resp, self.cond)
        save_train_curve(self.avg_train_mi_lb, self.avg_valid_mi_lb, figName)





