import numpy as np
from model import mine
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import randint
import os
from ..data import Gaussian, BiModal, Uniform
from ..utils import save_train_curve, mseEntropy, varEntropy, unifEntropy
from ..model import Mine, LinearReg, CL, AIC_TE, TableEntropy
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


    trainData, validData = train_val_split(data, mini_batch_size)  # ((N/2)X2), ((N/2)X2)

    # Mine
    mine = Mine(lr, mini_batch_size, patience)
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
        save_train_curve(tl[i], vl[i], filename)

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

def archive_20190411():
    cov = 0.99999
    batch_size = 300
    samples = 3
    input_arg = 'cov', cov, batch_size, samples
    ParallelWork(input_arg)




def ConditionSet(size, Resp, index):
    """[summary]
    
    Arguments:
        size {[int]} -- [size of powerset including resp]
        Resp {[int]} -- [index of Responce]
        index {[int]} -- [index in powerset]
    
    Returns:
        [list] -- [element in the subset excluding resp]
    """

    subset = TableEntropy.subsetVector(size - 1, index)
    subset = np.array(subset)
    cond = []
    for element in subset:
        if element >= Resp:
            element += 1
        cond.append(int(element))
    return cond


def ConditionIndex(size, Resp, cond):
    """[summary]
    
    Arguments:
        size {[int]} -- [size of cond]
        Resp {[int]} -- [responce in conditional entropy]
        cond {[masked numpy array or list]} -- [conditons in conditional entropy]
    
    Returns:
        [int] -- [index of conditional entropy in the Entropy table]
    """

    if (np.ma.is_masked(cond)):
        condSet = []
        for i in range(cond.size):
            if cond.mask[i] == False:
                if cond[i] > Resp:
                    condSet.append(int(cond[i] - 1))
                else:
                    condSet.append(int(cond[i]))
        # condSet = np.array(condSet)
        sizeUnmasked = int(size - np.ma.count_masked(cond))
        indexInResp = TableEntropy.subsetIndex(sizeUnmasked, condSet)
    else:
        condSet = np.zeros(cond.shape)
        for i in range(condSet.size):
            if cond[i] > Resp:
                condSet[i] = int(cond[i] - 1)
            else:
                condSet[i] = int(cond[i])
        indexInResp = TableEntropy.subsetIndex(size, condSet.tolist())
    return indexInResp

def example_LinReg_MMI():
    n_samples = 500
    n_test = 6
    cv_fold = 2
    for n_var in range(3, 3+n_test):
        X = Uniform(n_samples, n_var)
        data = X.data #n_sample * n_var
        cond_ent_LinReg = LinearReg(cv_fold).predict_Cond_Entropy(data)

        #build Entropy Table
        num_subset = 2**n_var
        ET_LinReg = -1*np.ones(num_subset)
        print ("resp, condInd, subset")
        for i in range(num_subset):
            ET_LinReg[i] = 0
            #compute entropy for all combination
            subset = np.ma.array(TableEntropy.subsetVector(n_var, i), mask=False)
            for j in range(subset.size):
                resp = subset[j]
                subset.mask[j] = True
                # if subset.size < n_var:
                condInd = int(ConditionIndex(subset.size, resp, subset))
                print (resp, condInd, subset)
                ET_LinReg[i] += cond_ent_LinReg[resp, condInd]
                ET_LinReg[i] += ET_LinReg[condInd]
                subset.mask[j] = False
            if subset.size > 0:
                ET_LinReg[i] /= subset.size
        
        psp = AIC_TE(ET_LinReg.tolist(), n_var)
        psp_gamma = np.array(psp.getCriticalValues())
        print (psp_gamma)
        print ("n_var = {0}".format(n_var))
        print ()

def computeMMI(EntropyTable, numVar, prefix="", log=True):
    """[summary]
    
    Arguments:
        EntropyTable {[numpy array]} -- [numVar X 2**(numVar-1)]
        numVar {[int]} -- [number of variable]
    
    Keyword Arguments:
        prefix {str} -- [prefix of log file] (default: {""})
        log {bool} -- [enable logging] (default: {True})
    
    Returns:
        [1 dim numpy array] -- [critical values of AIC]
    """

    #compute MMI
    psp = AIC_TE(EntropyTable.tolist(), numVar)

    while psp.agglomerate(1e-8, 1e-10):
        P = psp.getPartition(-1)
        CriticalValues = psp.getCriticalValues()
        P_np = np.array(P)
        print (P_np)
        CV = np.array(CriticalValues)
        print (CV)

    #Save result to files
    psp_gamma = np.array(psp.getCriticalValues())
    if log == True:
        np.savetxt("{0}psp_gamma.txt".format(prefix), psp_gamma)
        for gamma in psp_gamma:
            clusters = []
            for cluster in psp.getPartition(gamma):
                clusters.append(np.array(cluster))
            clusters = np.array(clusters)
            np.savetxt("{0}psp_clusters_gamma={1}.txt".format(prefix, gamma), clusters, fmt='%s')
    return psp_gamma


if __name__ == "__main__":
    n_samples = 500
    batch_size = 250
    n_test = 5
    lr = 1e-4
    patience = 10
    high = 1.0
    low = 0.0

    cv_fold = 2

    prefix_name = "Output/{0}/".format(datetime.now())
    os.mkdir(prefix_name)

    logName = "{0}parameter.log".format(prefix_name)
    log = open(logName, "w")
    log.write("n_sample={0}".format(n_samples))
    log.write("batch_size={0}".format(batch_size))
    log.write("n_test={0}".format(n_test))
    log.write("lr={0}".format(lr))
    log.write("patience={0}".format(patience))
    log.write("high={0}".format(high))
    log.write("low={0}".format(low))
    log.write("cvfold={0}".format(cv_fold))
    log.close()

    MMI_mine = []
    MMI_LinReg = []
    VAR = np.arange(3, 3+n_test)
    for n_var in VAR:
        #make loop specific folder
        prefix_name_loop = "{0}n_var={1}/".format(prefix_name, n_var)
        os.mkdir(prefix_name_loop)
        pn_mine = "{0}mine_".format(prefix_name_loop)
        pn_LinReg = "{0}LinReg_".format(prefix_name_loop)


        X = Uniform(n_samples, n_var, high=high, low=low)
        data = X.data #n_sample * n_var
        trainData, validData = train_val_split(data, batch_size=batch_size)

        #Current python wrapper dun support numpy.int64 as arg
        n_var = int(n_var)

        cond_ent_LinReg = LinearReg(cv_fold).predict_Cond_Entropy(data)

        numComb = 2**(n_var-1)
        cond_ent_mine = -1*np.ones((n_var, numComb))
        for Resp in range(n_var):
            cond_ent_mine[Resp, 0] = unifEntropy(0, high, low)
            for sI in range(1, numComb):
                subset = ConditionSet(n_var, Resp, sI)
                mine = Mine(lr, batch_size, patience, resp=Resp, cond=subset, prefix=pn_mine)
                mine.fit(train_data=trainData, val_data=validData)
                mine_est = mine.predict(validData)
                mine.savefig()
                cond_ent_mine[Resp, sI] = mine_est + cond_ent_mine[Resp, 0]

        #build Entropy Table
        num_subset = 2**n_var
        ET_mine = -1*np.ones(num_subset)
        ET_LinReg = -1*np.ones(num_subset)
        print ("resp, condInd, subset")
        for i in range(num_subset):
            ET_mine[i] = 0
            ET_LinReg[i] = 0
            #compute entropy for all combination
            sv = TableEntropy.subsetVector(n_var, i)
            subset = np.ma.array(sv, mask=False)
            for j in range(subset.size):
                resp = subset[j]
                subset.mask[j] = True
                # if subset.size < n_var:
                condInd = int(ConditionIndex(subset.size, resp, subset))
                print (resp, condInd, subset)
                ET_mine[i] += cond_ent_mine[resp, condInd]
                ET_mine[i] += ET_mine[condInd]
                ET_LinReg[i] += cond_ent_LinReg[resp, condInd]
                ET_LinReg[i] += ET_LinReg[condInd]
                subset.mask[j] = False
            if subset.size > 0:
                ET_mine[i] /= subset.size
                ET_LinReg[i] /= subset.size
        
        #compute MMI
        psp_gamma_mine = computeMMI(ET_mine, n_var, prefix=pn_mine)
        MMI_mine.append(psp_gamma_mine[-1])
        psp_gamma_LinReg = computeMMI(ET_LinReg, n_var, prefix=pn_LinReg)
        MMI_LinReg.append(psp_gamma_LinReg[-1])

        #Save result to files
        ET_mine = np.array(ET_mine)
        np.savetxt("{0}EntropyTable.txt".format(pn_mine), ET_mine)
        cond_ent_mine = np.array(cond_ent_mine)
        np.savetxt("{0}Cond_Ent.txt".format(pn_mine), cond_ent_mine)
        ET_LinReg = np.array(ET_LinReg)
        np.savetxt("{0}EntropyTable.txt".format(pn_LinReg), ET_LinReg)
        cond_ent_LinReg = np.array(cond_ent_LinReg)
        np.savetxt("{0}Cond_Ent.txt".format(pn_LinReg), cond_ent_LinReg)
        print ()
    

    #Save results
    MMI_mine = np.array(MMI_mine)
    np.savetxt("{0}MMI_mine.txt".format(prefix_name), MMI_mine)
    MMI_LinReg = np.array(MMI_LinReg)
    np.savetxt("{0}MMI_LinReg.txt".format(prefix_name), MMI_LinReg)

    #Save fig
    figName = "{0}plotMMI.png".format(prefix_name)
    fig,ax = plt.subplots()
    ax.scatter(VAR, MMI_mine, c='b', label='MINE')
    ax.scatter(VAR, MMI_LinReg, c='g', label='Regessor')
    plt.xlabel('# variables')
    plt.ylabel('MMI')
    ax.legend()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()