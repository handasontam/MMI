import numpy as np
from .model import mine
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import randint
import os
from .utils import save_train_curve
# from model import Mine, LinearReg, Kraskov
from joblib import Parallel, delayed
from . import settings
from tqdm import tqdm

def saveResultsFig(results_dict, experiment_path=""):
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

    for _, (model_name, dataset_results) in enumerate(results_dict.items()):
        for row_id, (dataset_name, results) in enumerate(dataset_results.items()):
            color = settings.model[model_name]['color']
            xs = [x for x, y in results]
            ys = [y for x, y in results]
            if n_datasets > 1:
                axe = axes[row_id]
            else:
                axe = axes
            axe.scatter(xs, ys, edgecolors=color, facecolors='none', label=model_name)
            axe.set_xlabel(settings.data[dataset_name]['varying_param_name'])
            axe.set_ylabel('MI')
            axe.set_title(dataset_name)
            axe.legend()
    figName = os.path.join("MI", experiment_path)
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

def get_estimation(data_model, data_name, varying_param, experiment_path):
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

    prefix_name_loop = os.path.join(experiment_path, "{}_{}={}/".format(data_model.name, data_model.varName, data_model.varValue))
    if not os.path.exists(prefix_name_loop):
        os.makedirs(prefix_name_loop)

    # Fit Algorithm
    for model_name, model in settings.model.items():
        # For plotting extra figure inside the training
        model['model'].model_name = model_name
        model['model'].prefix = os.path.join(prefix_name_loop, model_name)
        os.makedirs(model['model'].prefix)
        model['model'].paramName = data_model.varName
        model['model'].paramValue = data_model.varValue
        model['model'].ground_truth = ground_truth
        mi_estimation = model['model'].predict(data)

        # Save Results
        results[model_name] = mi_estimation
    
    # Ground Truth
    results['Ground Truth'] = ground_truth

    return results, data_name, varying_param

def plot(experiment_path):
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
    
    # # Main Loop
    r = Parallel(n_jobs=settings.cpu)(delayed(get_estimation)(data['model'](**kwargs), data_name, kwargs[data['varying_param_name']], experiment_path) 
                                                                for data_name, data in tqdm(settings.data.items())
                                                                for kwargs in tqdm(data['kwargs']))
    for (aggregate_result, data_name, varying_param) in r:
        for model_name, mi_estimate in aggregate_result.items():
            results[model_name][data_name].append((varying_param, mi_estimate))
    # Plot and save
    saveResultsFig(results, experiment_path=experiment_path)

    return 0


if __name__ == "__main__":
    # prompt
    experiment_name = input('Please enter the experiment name: ')
    experiment_path = os.path.join(settings.output_path, experiment_name)
    while True:
        if os.path.exists(experiment_path):
            experiment_name = input('experiment - \"{}\" already exists! Please reenter the expereiment name: '.format(experiment_name))
            experiment_path = os.path.join(settings.output_path, experiment_name)
        else:
            os.makedirs(experiment_path)
            print('Output will be saved into {}'.format(experiment_path))
            break
    plot(experiment_path)