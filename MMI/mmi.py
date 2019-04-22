import numpy as np
from .IC.AIC import CL, AIC_TE, TableEntropy
from . import MMI_config
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class MMI:
    def __init__(self, numRV, prefix="", log=True):
        self.numRV = int(numRV)
        self.prefix = prefix
        self.log = log

    def index2cond(self, Resp, index):
        subset = TableEntropy.subsetVector(self.numRV - 1, index)
        subset = np.array(subset)
        cond = []
        for element in subset:
            if element >= Resp:
                element += 1
            cond.append(int(element))
        return cond

    def cond2index(self, Resp, cond):
        sizeCond = self.numRV - 1
        if (np.ma.is_masked(cond)):
            condSet = []
            for i in range(cond.size):
                if cond.mask[i] == False:
                    if cond[i] > Resp:
                        condSet.append(int(cond[i] - 1))
                    else:
                        condSet.append(int(cond[i]))
            indexInResp = TableEntropy.subsetIndex(sizeCond, condSet)
        else:
            condSet = np.zeros(cond.shape, dtype=int)
            for i in range(condSet.size):
                if cond[i] > Resp:
                    condSet[i] = int(cond[i] - 1)
                else:
                    condSet[i] = int(cond[i])
            indexInResp = TableEntropy.subsetIndex(sizeCond, condSet.tolist())
        return indexInResp

    def set2index(self, cond):
        size = self.numRV
        if (np.ma.is_masked(cond)):
            Set = []
            for i in range(cond.size):
                if cond.mask[i] == False:
                    Set.append(int(cond[i]))
            index = TableEntropy.subsetIndex(size, Set)
        else:
            Set = np.zeros(cond.shape, dtype=int)
            for i in range(Set.size):
                Set[i] = int(cond[i])
            index = TableEntropy.subsetIndex(size, Set.tolist())
        return index

    # def computeMMI(self, subsetEntropyTable):
    #     """[summary]
        
    #     Arguments:
    #         subsetEntropyTable {[np array, list]} -- [numSubsetEntropy]
        
    #     Raises:
    #         TypeError -- [when subsetEntropyTable is neither list and np array]
        
    #     Returns:
    #         [np array] -- [gamma value of clusters]
    #     """

    #     if np.ndarray == type(subsetEntropyTable):
    #         subsetEntropyTable = subsetEntropyTable.tolist()
    #     elif list != type(subsetEntropyTable):
    #         errMsg = "subsetEntropyTable[{0}] should be list or np array".format(type(subsetEntropyTable))
    #         raise TypeError(errMsg)
    #     #compute MMI
    #     psp = AIC_TE(subsetEntropyTable, self.numRV)

    #     n_iter = 0
    #     while psp.agglomerate(1e-8, 1e-10):
    #         #Save result to files
    #         if self.log == True:
    #             psp_gamma = np.array(psp.getCriticalValues())
    #             np.savetxt("{0}psp_{1}_gamma.txt".format(self.prefix, n_iter), psp_gamma)
    #             for gamma in psp_gamma:
    #                 clusters = []
    #                 for cluster in psp.getPartition(gamma):
    #                     clusters.append(np.array(cluster))
    #                 clusters = np.array(clusters)
    #                 np.savetxt("{0}psp_{1}_clusters_gamma={2}.txt".format(self.prefix, n_iter, gamma), clusters, fmt='%s')
    #         n_iter +=1

    #     #Save result to files
    #     psp_gamma = np.array(psp.getCriticalValues())
    #     if self.log == True:
    #         np.savetxt("{0}psp_gamma.txt".format(self.prefix), psp_gamma)
    #         for gamma in psp_gamma:
    #             clusters = []
    #             for cluster in psp.getPartition(gamma):
    #                 clusters.append(np.array(cluster))
    #             clusters = np.array(clusters)
    #             np.savetxt("{0}psp_clusters_gamma={1}.txt".format(self.prefix, gamma), clusters, fmt='%s')
    #     return psp_gamma

    def condEntTables2subsetEntTables(self, condEntTables, verbose=False):
        """[summary]
        
        Arguments:
            condEntTables {[dict] or [np array]} -- [numModel X numVariable X numConditions]
        
        Keyword Arguments:
            verbose {bool} -- [debug subset and index] (default: {False})
        
        Raises:
            TypeError -- [when condEntTables is neither dict and np array]
        
        Returns:
            [dict] or [np array] -- [numModel X numSubsetEntropy]
        """

        if np.ndarray == type(condEntTables):
            num_tables = condEntTables.shape[0]
            num_subset = 2**self.numRV
            subsetEnt = np.zeros((num_tables, num_subset))
        elif dict == type(condEntTables):
            num_tables = len(condEntTables)
            num_subset = 2**self.numRV
            subsetEnt = dict()
            for model_name in condEntTables.keys():
                subsetEnt[model_name] = np.zeros(num_subset)
        else:
            errMsg = "condEntTable[{0}] should be np array or dict".format(type(condEntTables))
            raise TypeError(errMsg)

        if verbose:
            print ("resp\tcondInd\tsubset")
        for i in range(num_subset):
            #compute entropy for all combination
            subset = np.ma.array(TableEntropy.subsetVector(self.numRV, i), mask=False)
            for j in range(subset.size):
                resp = subset[j]
                subset.mask[j] = True
                condInd = int(self.cond2index(resp, subset))
                setInd = int(self.set2index(subset))
                if verbose:
                    print ("{0}\t{1}\t{2}".format(resp, condInd, subset))
                if np.ndarray == type(condEntTables):
                    subsetEnt[:,i] += condEntTables[:, resp, setInd]
                    subsetEnt[:,i] += subsetEnt[:,condInd]
                elif dict == type(condEntTables):
                    for model_name in condEntTables.keys():
                        subsetEnt[model_name][i] += condEntTables[model_name][int(resp), condInd]
                        subsetEnt[model_name][i] += subsetEnt[model_name][setInd]
                subset.mask[j] = False
            if subset.size > 0:
                if np.ndarray == type(condEntTables):
                    subsetEnt[:,i] /= subset.size
                elif dict == type(condEntTables):
                    for model_name in condEntTables.keys():
                        subsetEnt[model_name][i] /= subset.size
        if self.log == True:
            if np.ndarray == type(condEntTables):
                np.savetxt("{0}condEntropyTable.txt".format(self.prefix), condEntTables)
                np.savetxt("{0}subsetEntropyTable.txt".format(self.prefix), subsetEnt)
            elif dict == type(condEntTables):
                for model_name in condEntTables.keys():
                    np.savetxt("{0}{1}_condEntropyTable.txt".format(self.prefix, model_name), condEntTables[model_name])
                    np.savetxt("{0}{1}_subsetEntropyTable.txt".format(self.prefix, model_name), subsetEnt[model_name])
        return subsetEnt

    def condEntTable2MMI(self, condEntTables, verbose=False):
        """[summary]
        
        Arguments:
            condEntTables {[dict]} -- [numModel X numVariable X numConditions]
        
        Keyword Arguments:
            verbose {bool} -- [debug subset and index] (default: {False})

        Raises:
            TypeError -- [when condEntTables is neither dict and np array]
        
        Returns:
            [dict] -- [numModel, it contain MMI of each model]
        """

        if dict != type(condEntTables):
            raise TypeError("condEntTables[{0}] should be dict".format(type(condEntTables)))
        subsetEntTables = self.condEntTables2subsetEntTables(condEntTables)
        results = dict()
        for model_name in MMI_config.model.keys():
            if np.ndarray == type(subsetEntTables[model_name]):
                subsetEntTables[model_name] = subsetEntTables[model_name].tolist()
            elif list != type(subsetEntTables[model_name]):
                errMsg = "subsetEntTables[model_name][{0}] should be list or np array".format(type(subsetEntTables[model_name]))
                raise TypeError(errMsg)
            #compute MMI
            psp = AIC_TE(subsetEntTables[model_name], self.numRV)

            n_iter = 0
            while psp.agglomerate(1e-8, 1e-10):
                #Save result to files
                if self.log == True:
                    psp_gamma = np.array(psp.getCriticalValues())
                    np.savetxt("{0}{2}_psp_{1}_gamma_.txt".format(self.prefix, n_iter, model_name), psp_gamma)
                    for gamma in psp_gamma:
                        clusters = []
                        for cluster in psp.getPartition(gamma):
                            clusters.append(np.array(cluster))
                        clusters = np.array(clusters)
                        np.savetxt("{0}{3}_psp_{1}_clusters_gamma={2}.txt".format(self.prefix, n_iter, gamma, model_name), clusters, fmt='%s')
                n_iter +=1

            #Save result to files
            psp_gamma = np.array(psp.getCriticalValues())
            if self.log == True:
                np.savetxt("{0}{1}_psp_gamma.txt".format(self.prefix, model_name), psp_gamma)
                for gamma in psp_gamma:
                    clusters = []
                    for cluster in psp.getPartition(gamma):
                        clusters.append(np.array(cluster))
                    clusters = np.array(clusters)
                    np.savetxt("{0}{2}_psp_clusters_gamma={1}.txt".format(self.prefix, gamma, model_name), clusters, fmt='%s')
            # Save Results
            results[model_name] = psp_gamma[0]
        return results


def saveResultsFig(results_dict, prefix=""):
    """
    
    Arguments:
    # results_dict example: 
    # {
    #     'Ground Truth': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],  # [(rho, MI), (rho2, MI_2), ...]
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     'Linear Regression': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     ...
    # }
    """
    # initialize ground truth color
    MMI_config.model['Ground Truth'] = {'color': 'red'}
    
    n_datasets = MMI_config.n_datasets
    # n_columns = MMI_config.n_columns + 1  # 0 to N_Column for visualizing the data, last column for the MI estimate plot

    fig, axes = plt.subplots(nrows=n_datasets, ncols=1, figsize=(12,8))

    for column_id, (model_name, dataset_results) in enumerate(results_dict.items()):
        for row_id, (dataset_name, results) in enumerate(dataset_results.items()):
            color = MMI_config.model[model_name]['color']
            xs = [x for x, y in results]
            ys = [y for x, y in results]
            if n_datasets==1:
                axes.scatter(xs, ys, edgecolors=color, facecolors='none', label=model_name)
                axes.set_xlabel(MMI_config.data[dataset_name]['varying_param_name'])
                axes.set_ylabel('MI')
                axes.set_title(dataset_name)
                axes.legend()
            else:
                axes[row_id].scatter(xs, ys, edgecolors=color, facecolors='none', label=model_name)
                axes[row_id].set_xlabel(MMI_config.data[dataset_name]['varying_param_name'])
                axes[row_id].set_ylabel('MI')
                axes[row_id].set_title(dataset_name)
                axes[row_id].legend()
    figName = "{0}MI".format(prefix)
    fig.savefig(figName, bbox_inches='tight')
    plt.close()


def get_estimation_MMI(data_model, varying_param, prefix=""):
    """
    Returns: results, example:
                
        results example: 
        {
            'Ground Truth': 0.5, 
            'Linear Regression': 0.4, 
            'SVM': 0.4, ...
        }
    """
    # results = dict()
    data = data_model.data

    prefix_name_loop = "{0}n_var={1}/".format(MMI_config.prefix_name, data.shape[1])
    os.mkdir(prefix_name_loop)

    # Fit Algorithm
    condEntTables = dict()
    for model_name, model in MMI_config.model.items():
        condEntTables[model_name] = model['model'].predict_Cond_Entropy(data)
    mmi = MMI(data.shape[1], prefix=prefix_name_loop)
    results = mmi.condEntTable2MMI(condEntTables)
    
    # Ground Truth
    # ground_truth = data_model.ground_truth
    # results['Ground Truth'] = ground_truth

    return results, varying_param

def plotMMI():
    prefix_name = MMI_config.prefix_name
    os.mkdir(prefix_name)
    results = dict()
    # results['Ground Truth'] = dict()
    for model_name in MMI_config.model.keys():
        results[model_name] = dict()
        for data_name in MMI_config.data.keys():
            results[model_name][data_name] = []
            # results['Ground Truth'][data_name] = []
    
    # Main Loop
    for data_name, data in tqdm(MMI_config.data.items()):
        data_model = data['model']
        varying_param_name = data['varying_param_name']
        r = Parallel(n_jobs=MMI_config.cpu)(delayed(get_estimation_MMI)(data_model(**kwargs), kwargs[varying_param_name]) for kwargs in tqdm(data['kwargs']))
        for (aggregate_result, varying_param) in r:
            for model_name, mi_estimate in aggregate_result.items():
                results[model_name][data_name].append((varying_param, mi_estimate))
    # Plot and save
    saveResultsFig(results, prefix=prefix_name)