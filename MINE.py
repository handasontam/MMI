#!/usr/bin/env python
# coding: utf-8

# In[5]:


from pytorchtools import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

class Mine(nn.Module):
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

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint))
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
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

def valid_mine(batch, mine_net,  ma_et, ma_rate=0.01):
    joint , marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint))
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    return loss
    

def create_dataset(data, batch_size=100):
    if data.shape[0] >= batch_size * 2:
        partSize = int(data.shape[0]/2)
        indices = list(range(data.shape[0]))
        np.random.shuffle(indices)
        valid_idx = indices[:partSize]
        train_idx = indices[partSize:]
        train_data = data[train_idx]
        valid_data = data[valid_idx]
        return train_data, valid_data
    
def sample_batch(data, resp, cond, batch_size=100, sample_mode='joint', randomJointIdx=True):
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch_joint = data[index]
    if randomJointIdx == True:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        if data.shape[1] == 2:
            batch_mar = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                         data[marginal_index][:,1].reshape(-1,1)],
                                       axis=1)
        else:
            batch_mar = np.concatenate([data[joint_index][:,resp].reshape(-1,1),
                                         data[marginal_index][:,cond].reshape(-1,len(cond))],
                                       axis=1)
    else:
        marginal_index = np.random.choice(range(batch_joint.shape[0]), size=batch_size, replace=False)
        if data.shape[1] == 2:
            batch_mar = np.concatenate([batch_joint[:,0].reshape(-1,1),
                                         batch_joint[marginal_index][:,1].reshape(-1,1)],
                                       axis=1)
        else:
            batch_mar = np.concatenate([batch_joint[:,resp].reshape(-1,1),
                                         batch_joint[marginal_index][:,cond].reshape(-1,len(cond))],
                                       axis=1)
    if (type(cond)==list):
        whole = cond.copy()
        whole.append(resp)
        batch_joint = batch_joint[:,whole]
    return batch_joint, batch_mar

def train(data, mine_net,mine_net_optim, resp=0, cond=1, batch_size=100          , iter_num=int(1e+3), log_freq=int(1e+2)          , avg_freq=int(1e+1), verbose=True, patience=20, index=0):
    # data is x or y
    result = list()
    ma_et = 1.
    
    #Early Stopping
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    earlyStop = EarlyStopping(patience=patience, verbose=True)
    trainData, validData = create_dataset(data, batch_size)
    for i in range(iter_num):
        #get train data
        batchTrain = sample_batch(trainData,resp, cond, batch_size=batch_size, randomJointIdx=False)
        mi_lb, ma_et, lossTrain = learn_mine(batchTrain, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        train_losses.append(lossTrain.item())
        if verbose and (i+1)%(log_freq)==0:
            print(result[-1])
        
        batchValid = sample_batch(validData,resp, cond, batch_size=batch_size)
        mi_lb_valid = valid_mine(batchValid, mine_net, ma_et)
        valid_losses.append(mi_lb_valid.item())
        
        if (i+1)%(avg_freq)==0:
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print_msg = "[{0}/{1}] train_loss: {2} valid_loss: {3}".format(i, iter_num, train_loss, valid_loss)
            print (print_msg)

            train_losses = []
            valid_losses = []

            earlyStop(valid_loss, mine_net, index=index)
            if (earlyStop.early_stop):
                print("Early stopping")
                break
    
    ch = "checkpoint_{0}.pt".format(index)
    mine_net.load_state_dict(torch.load(ch))#'checkpoint.pt'))
    return result, mine_net, avg_train_losses, avg_valid_losses

def ma(a, window_size=100):
    if len(a)<=window_size+1:
        return [np.mean(a)]
    else:
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

def visualizeTrainLogAndSave(train_loss, valid_loss, figName):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import randint
import DiscreteCondEnt as DC

from sklearn.linear_model import LinearRegression

def varEntropy(y):
    return np.log(np.var(y)*np.pi*2)/2

from sklearn.metrics import mean_squared_error
def MSEscorer(clf, X, y):
    y_est = clf.predict(X)
    return np.log(mean_squared_error(y, y_est)*np.pi*2)/2


linReg = LinearRegression()

def worker_Train_Mine_cov(input_arg):
    cov, MINEsize, index= input_arg
    MINEsize = int(float(MINEsize))
    cov = float(cov)
    index = int(float(index))
    Size = int(MINEsize*10)

    data = 'gaussian'
    if 'bimodel' == data:
        x1 = 0.5 * np.transpose(np.random.multivariate_normal( mean=[0,0], cov=[[1,cov],[cov,1]], size=Size))
        x2 = 0.5 * np.transpose(np.random.multivariate_normal( mean=[0,0], cov=[[1,-1*cov],[-1*cov,1]], size=Size))
        x = x1 + x2
    elif 'gaussian' == data:
        x = np.transpose(np.random.multivariate_normal( mean=[0,0], cov=[[1,cov],[cov,1]], size = MINEsize * 10))
    CVFold = 3
    DE = DC.computeEnt(x, linReg, MSEscorer, varEntropy, CVFold)
    MI = DE[1,0] + DE[0,0] - DE[0,1] - DE[1,1]
    MI = MI/2
    REG = MI
    GT = -0.5*np.log(1-cov*cov)
    mine_net = Mine()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-6)
    result, mine_net,tl ,vl = train(np.transpose(x),mine_net,mine_net_optim, verbose=False, batch_size=MINEsize, patience=10, index=index)
    result_ma = ma(result)
    MINE = result_ma[-1]
    return cov, MINE, REG, GT, MINEsize, tl, vl

from multiprocessing.dummy import Pool as ThreadPool

def ParallelWork(input_arg):
    mode, Cov, Size, numThreads, prefix_name = input_arg

    numThreads = int(numThreads)
    if (numThreads < 1):
        numThreads = 1
    if ('cov' == mode):
        cov = 1 - 0.1**np.arange(numThreads)
        size = int(Size)*np.ones(numThreads)
    elif ('size' == mode):
        size = int(2)**np.arange(numThreads)*50
        cov = Cov*np.ones(numThreads)

    index = np.arange(numThreads)

    inputArg = np.concatenate((cov[:,None],size[:,None]),axis=1)
    inputArg = np.concatenate((inputArg,index[:,None]),axis=1)
    inputArg = inputArg.tolist()
    pool = ThreadPool(numThreads)
    results = pool.map(worker_Train_Mine_cov, inputArg)
    pool.close()
    pool.join()

    results = np.array(results)
    COV2 = results[:,0]
    MINE2 = results[:,1]
    Reg2 = results[:,2]
    GT2 = results[:,3]
    size2 = results[:,4]
    tl = results[:,5]
    vl = results[:,6]

    for i in range(numThreads):
        filename = "{2}MINE_Train_Fig_cov={0}_size={1}.png".format(COV2[i],size2[i],prefix_name)
        visualizeTrainLogAndSave(tl[i], vl[i], filename)

    if 'cov' == mode:
        filename = "{0}MINE-{2}_size={1}.png".format(prefix_name, Size, mode)
        xlabel = "covariance"
        saveResultFig(filename, GT2, Reg2, MINE2, COV2, xlabel)
    elif 'size' == mode:
        filename = "{0}MINE-{2}_cov={1}.png".format(prefix_name, Cov,mode)
        xlabel = "batch size"
        saveResultFig(filename, GT2, Reg2, MINE2, size2, xlabel)


    if 'cov' == mode:
        COV2 = COV2.astype(float)
        COV2 = np.log(np.ones(COV2.shape)-COV2)
        filename = "{0}MINE-{2}_log_size={1}.png".format(prefix_name, Size, mode)
        xlabel = "ln(1 - covariance)"
        saveResultFig(filename, GT2, Reg2, MINE2, COV2, xlabel)
    elif 'size' == mode:
        size2 = size2.astype(float)
        size2 = np.log(size2)
        filename = "{0}MINE-{2}_log_cov={1}.png".format(prefix_name, Cov, mode)
        xlabel = "ln(size)"
        saveResultFig(filename, GT2, Reg2, MINE2, size2, xlabel)

    plt.close('all')
    return results

def saveResultFig(figName, GT, Reg, MINE, COV, xlabel):
    fig,ax = plt.subplots()
    ax.scatter(COV, MINE, c='b', label='MINE')
    # ax.scatter(COV, Reg, c='r', label='Regressor')
    # ax.scatter(COV, GT, c='g', label='Ground Truth')
    plt.xlabel(xlabel)
    plt.ylabel('mutual information')
    ax.legend()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()



# In[16]:
if __name__ == "__main__":
    from datetime import datetime
    import os
    folderName = "data/{0}/".format(datetime.now())
    os.mkdir(folderName)
    cov = 0
    batch_size = 400
    samples = 6
    input_arg = 'size', cov, batch_size, samples, folderName
    ParallelWork(input_arg)