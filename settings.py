# import model
from .model.linear_regression import LinearReg
from .model.mine import Mine
from .model.kraskov import Kraskov
from .model.cart_regression import cartReg

from .model.ShannonKDE import ShanKDE
from .model.hellingerDiv import hellingerDiv
from .model.tsallisDiv import tsallisDiv
from .model.chiSqDiv import chiSqDiv
from .model.renyiDiv import renyiDiv
from .model.klDiv import klDiv
from .model.condShannonEntropy import condShanEnt

from .data.bimodal import BiModal
from .data.gaussian import Gaussian
from .data.uniform import Uniform
import math
import os
from datetime import datetime

cpu = 1

batch_size=300

prefix_name = "MMI/Output/main_{0}/".format(datetime.now())

# ground truth is plotted in red
model = {
    'Linear Regression': {  # model name, for plotting the legend
        'model': LinearReg(  # initialize the object
            cvFold=3
        ), 
        'color': 'blue'  # for plotting
    }, 
    'Kraskov': {
        'model': Kraskov(
            discrete_features='auto', 
            n_neighbors=3, 
            random_state=None
        ), 
        'color': 'green'
    }, 
    # 'LOO Shannon KDE': {
    #     'model': ShanKDE(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'magenta'
    # },
    # 'LOO hellingerDiv': {
    #     'model': hellingerDiv(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'cyan'
    # }, 
    # 'LOO tsallisDiv': {
    #     'model': tsallisDiv(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'yellow'
    # }, 
    # 'LOO chiSqDiv': {
    #     'model': chiSqDiv(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'black'
    # }, 
    # 'LOO renyiDiv': {
    #     'model': renyiDiv(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'green'
    # }, 
    # 'LOO klDiv': {
    #     'model': klDiv(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'orange'
    # }, 
    # 'LOO condShanEnt': {
    #     'model': condShanEnt(
    #         numPart='loo', 
    #         numAvgPart=1, 
    #         correctBound=False, 
    #         Low=1e-5, 
    #         Upp=math.inf, 
    #         doAsympAnalysis=False,
    #         alpha=0.5
    #     ), 
    #     'color': 'pink'
    # }, 
    'Cart Reg': {
        'model': cartReg(
            cvFold=3
        ), 
        'color': 'pink'
    },
    'MINE_direct': {
        'model': Mine(
            lr=1e-4, 
            batch_size=batch_size, 
            patience=int(20), 
            iter_num=int(2e+3), 
            log_freq=int(100), 
            avg_freq=int(10), 
            ma_rate=0.01, 
            verbose=False,
            prefix=prefix_name,
            marginal_mode='shuffle',
            objName='Dir'
        ), 
        'color': 'orange'
    },
    'MINE_entropy': {
        'model': Mine(
            lr=1e-3, 
            batch_size=batch_size, 
            patience=int(20), 
            iter_num=int(2e+4), 
            log_freq=int(100), 
            avg_freq=int(10), 
            ma_rate=0.01, 
            verbose=False,
            prefix=prefix_name,
            marginal_mode='unif',
            objName='Ent'
        ), 
        'color': 'purple'
    }
}

n_samples = batch_size * 20
rhos = [0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999 ]
variables = [3, 4, 5, 6, 7]
widths = list(range(10))

varName = 'correlation'

data = {
    'BiModal': {
        'model': BiModal,
        'kwargs': [  # list of params
            {
                'n_samples':n_samples, 
                'mean1':0, 
                'mean2':0, 
                'rho1': rho, 
                'rho2': -rho,
                'varName': varName,
                'varValue': rho
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    'Gaussian': {
        'model': Gaussian, 
        'kwargs': [
            {
                'n_samples':n_samples, 
                'mean1':0, 
                'mean2':0, 
                'rho': rho,
                'varName': varName,
                'varValue': rho
            } for rho in rhos
        ], 
        'varying_param_name': 'rho', 
        'x_axis_name': varName, 
    },
    # {
    #     'name': 'Examples', 
    #     'model': XX(
    #         n_samples=XX
    #         rho=XX
    #     )
    # }, 
}


n_datasets = len(data)
# n_columns = max([len(rhos), len(widths)])