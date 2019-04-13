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
import data
from utils import save_train_curve
# from model import Mine, LinearReg, Kraskov
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed
import settings
from tqdm import tqdm

def saveResultsFig(results_dict):
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
    settings.model['Ground Truth'] = {'color': 'red'}
    
    n_datasets = settings.n_datasets
    # n_columns = settings.n_columns + 1  # 0 to N_Column for visualizing the data, last column for the MI estimate plot

    fig, axes = plt.subplots(nrows=n_datasets, ncols=1, figsize=(12,8))

    for column_id, (model_name, dataset_results) in enumerate(results_dict.items()):
        for row_id, (dataset_name, results) in enumerate(dataset_results.items()):
            color = settings.model[model_name]['color']
            xs = [x for x, y in results]
            ys = [y for x, y in results]
            axes[row_id].scatter(xs, ys, edgecolors=color, facecolors='none', label=model_name)
            axes[row_id].set_xlabel(settings.data[dataset_name]['varying_param_name'])
            axes[row_id].set_ylabel('MI')
            axes[row_id].set_title(dataset_name)
            axes[row_id].legend()
    fig.savefig('MI', bbox_inches='tight')
    plt.close()

def get_estimation(data_model, varying_param):
    """
    Returns: results, example:
                
        results example: 
        {
            'Ground Truth': 0.5, 
            'Linear Regression': 0.4, 
            'SVM': 0.4, ...
        }
    """

    results = dict()
    data = data_model.data
    ground_truth = data_model.ground_truth

    # Fit Algorithm
    for model_name, model in settings.model.items():
        mi_estimation = model['model'].predict(data)

        # Save Results
        results[model_name] = mi_estimation
    
    # Ground Truth
    results['Ground Truth'] = ground_truth

    return results, varying_param

def plot():
    # Initialize the results dictionary

    # results example: 
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

    results = dict()
    results['Ground Truth'] = dict()
    for model_name in settings.model.keys():
        results[model_name] = dict()
        for data_name in settings.data.keys():
            results[model_name][data_name] = []
            results['Ground Truth'][data_name] = []
    
    # Main Loop
    for data_name, data in tqdm(settings.data.items()):
        data_model = data['model']
        varying_param_name = data['varying_param_name']
        r = Parallel(n_jobs=settings.cpu)(delayed(get_estimation)(data_model(**kwargs), kwargs[varying_param_name]) for kwargs in tqdm(data['kwargs']))
        for (aggregate_result, varying_param) in r:
            for model_name, mi_estimate in aggregate_result.items():
                results[model_name][data_name].append((varying_param, mi_estimate))
    # Plot and save
    saveResultsFig(results)


    return 0


if __name__ == "__main__":
    plot()