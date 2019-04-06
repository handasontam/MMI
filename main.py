import numpy as np
from model import mine
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import randint
import os
from data import Gaussian, BiModal 
from data.utils import train_val_split
from utils import visualizeTrainLogAndSave
from model import Mine, LinearReg
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool




def ma(a, window_size=100):
    if len(a)<=window_size+1:
        return [np.mean(a)]
    else:
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

def worker_Train_Mine_cov(input_arg):
    cov, mini_batch_size, index, prefix= input_arg
    cov = float(cov)
    index = int(float(index))
    mini_batch_size= int(float(mini_batch_size))
    SampleSize = int(mini_batch_size*10)
    
    dataType = 'bimodal'
    lr = 1e-3
    cvFold = 3
    patience = 10

    prefix_name = "{0}_{1}/".format(prefix, index)
    os.mkdir(prefix_name)

    logName = "{0}parameter.log".format(prefix_name)
    log = open(logName, "w")
    log.write("dataType={0}\n".format(dataType))
    log.write("cov={0}\n".format(cov))
    log.write("batch_size={0}\n".format(mini_batch_size))
    log.write("Sample_size={0}\n".format(SampleSize))
    log.write("lr={0}\n".format(lr))
    log.write("CV Folds={0}\n".format(cvFold))
    log.close

    if 'bimodal' == dataType:
        X = BiModal(n_samples=SampleSize, mean1=0, mean2=0, rho1=0.9, rho2=-0.9, mix=0.5)
    elif 'gaussian' == dataType:
        X = Gaussian(n_samples=SampleSize, mean=[0,0], covariance=cov)
    data = X.data  # 2 X N
    ground_truth = X.ground_truth  # float
    np.savetxt("{0}Sample.txt".format(prefix_name), data)


    trainData, validData = train_val_split(data, batch_size)  # ((N/2)X2), ((N/2)X2)

    # Mine
    mine = Mine(lr, batch_size, patience)
    mine.fit(train_data=trainData, val_data=validData)
    mine_est = mine.predict(validData)

    # Linear Regressor
    linear_reg_est = LinearReg(cvFold).predict(X.data)

    # result_ma = ma(result)
    # MINE = result_ma[-1]
    return cov, mine_est, linear_reg_est, ground_truth, mini_batch_size, mine.avg_train_losses, mine.avg_valid_losses

# 'cov', cov, batch_size, samples
def ParallelWork(input_arg):
    mode, Cov, Size, numThreads = input_arg
    
    numThreads = int(numThreads)
    if (numThreads < 1):
        numThreads = 1
    if ('cov' == mode):
        cov = 1 - 0.1**np.arange(numThreads)
        size = int(Size)*np.ones(numThreads)
    elif ('size' == mode):
        size = int(2)**np.arange(numThreads)*50
        cov = Cov*np.ones(numThreads)

    prefix_name = "data/{0}/".format(datetime.now())
    os.mkdir(prefix_name)

    logName = "{0}parameter.log".format(prefix_name)
    log = open(logName, "w")
    log.write("mode={0}\n".format(mode))
    log.write("#Thread={0}\n".format(numThreads))
    if 'cov' == mode:
        log.write("size={0}\n".format(Size))
        log.write("cov={0}\n".format(cov))
    elif 'size' == mode:
        log.write("size={0}\n".format(size))
        log.write("cov={0}\n".format(Cov))
    log.close()

    index = np.arange(numThreads)
    prefix_array = np.array([prefix_name for _ in range(numThreads)])
    inputArg = np.concatenate((cov[:,None],size[:,None]),axis=1)
    inputArg = np.concatenate((inputArg,index[:,None]),axis=1)
    inputArg = np.concatenate((inputArg,prefix_array[:,None]),axis=1)
    inputArg = inputArg.tolist()

    #Workers start working
    pool = ThreadPool(numThreads)
    results = pool.map(worker_Train_Mine_cov, inputArg)
    pool.close()
    pool.join()

    results = np.array(results)  # (n thread X 7)
    COV2 = results[:,0]
    np.savetxt("{0}cache_COV.txt".format(prefix_name), COV2)
    MINE2 = results[:,1]
    np.savetxt("{0}cache_MINE.txt".format(prefix_name), MINE2)
    Reg2 = results[:,2]
    np.savetxt("{0}cache_REG.txt".format(prefix_name), Reg2)
    GT2 = results[:,3]
    np.savetxt("{0}cache_GT.txt".format(prefix_name), GT2)
    size2 = results[:,4]
    np.savetxt("{0}cache_size.txt".format(prefix_name), size2)
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
    ax.scatter(COV, Reg, c='r', label='Regressor')
    ax.scatter(COV, GT, c='g', label='Ground Truth')
    plt.xlabel(xlabel)
    plt.ylabel('mutual information')
    ax.legend()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()



# In[16]:
if __name__ == "__main__":
    cov = 0.99999
    batch_size = 300
    samples = 3
    input_arg = 'cov', cov, batch_size, samples
    ParallelWork(input_arg)