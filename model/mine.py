import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from .pytorchtools import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split


def sample_joint_marginal(data, batch_size=100, randomJointIdx=True):
    """[summary]
    
    Arguments:
        data {[type]} -- [N X 2]
        resp {[int]} -- [description]
        cond {[np array]} -- [1 dimension]
    
    Keyword Arguments:
        batch_size {int} -- [description] (default: {100})
        randomJointIdx {bool} -- [description] (default: {True})
    
    Returns:
        [batch_joint] -- [batch size X 2]
        [batch_mar] -- [batch size X 2]
    """
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch_joint = data[index]
    joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch_mar = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                    data[marginal_index][:,1].reshape(-1,1)],
                                axis=1)
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
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(2e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, prefix="", verbose=True):
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience  # 20
        self.iter_num = iter_num  # 1e+3
        self.log_freq = int(log_freq)  # int(1e+2)
        self.avg_freq = avg_freq  # int(1e+1)
        self.ma_rate = ma_rate  # 0.01
        self.prefix = prefix
        self.verbose = verbose

    def fit(self, train_data, val_data):
        self.mine_net = MineNet()
        self.mine_net_optim = optim.Adam(self.mine_net.parameters(), lr=self.lr)
    
        randomJointIdx = False
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
            batchTrain = sample_joint_marginal(train_data, batch_size=self.batch_size, randomJointIdx=randomJointIdx)
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

                # print_msg = "[{0}/{1}] train_loss: {2} valid_loss: {3}".format(i, self.iter_num, train_loss, valid_loss)
                # print (print_msg)

                train_mi_lb = []
                valid_mi_lb = []

                earlyStop(valid_loss, self.mine_net)
                if (earlyStop.early_stop):
                    if self.verbose:
                        print("Early stopping")
                    break
        
        # ch = "{0}checkpoint.pt".format(self.prefix)
        # self.mine_net.load_state_dict(torch.load(ch))#'checkpoint.pt'))

    
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
        joint , marginal = sample_joint_marginal(X, batch_size=X.shape[0])
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




