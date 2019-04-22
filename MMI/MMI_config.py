# import model
# from model.linear_regression import LinearReg
# from model.mine import Mine
# from model.kraskov import Kraskov
# from data.bimodal import BiModal
# from data.gaussian import Gaussian
# from data.uniform import Uniform

from datetime import datetime

from ..model.linear_regression import LinearReg
from ..model.logistic_regression import LogisticReg
from ..model.mine import Mine
from ..model.kraskov import Kraskov
from ..model.cart_regression import cartReg
from ..model.gaussianEntropy import GaussianEntropy
from ..data.uniform import Uniform

cpu = 1

batch_size=300


n_samples = 500
n_test = 3

prefix_name = "MMI/Output/{0}/".format(datetime.now())

# import numpy as np
# k = 2
# sigma = 1
# n = 3

# A_GE = np.zeros([n,n+k])
# for i in range(A_GE.shape[0]):
#     A_GE[i, i%k] = 1
#     A_GE[i, i+k] = sigma

# Sigma_GE = np.matmul(A_GE, np.transpose(A_GE))


# ground truth is plotted in red
model = {
    # 'Gaussian Entropy': {
    #     'model': GaussianEntropy(
    #         Sigma_GE
    #     ), 
    #     'color': 'red'
    # },
    'Linear Regression': {  # model name, for plotting the legend
        'model': LinearReg(  # initialize the object
            cvFold=3
        ), 
        'color': 'blue'  # for plotting
    }, 
    'Cart Reg': {
        'model': cartReg(
            cvFold=3
        ), 
        'color': 'pink'
    }, 
    'Kraskov': {
        'model': Kraskov(
            discrete_features='auto', 
            n_neighbors=3, 
            random_state=None
        ), 
        'color': 'green'
    }, 
    # 'MINE_direct': {
    #     'model': Mine(
    #         lr=1e-3, 
    #         batch_size=batch_size, 
    #         patience=int(20), 
    #         iter_num=int(2e+4), 
    #         log_freq=int(100), 
    #         avg_freq=int(10), 
    #         ma_rate=0.01, 
    #         verbose=False,
    #         prefix=prefix_name,
    #         marginal_mode='shuffle'
    #     ), 
    #     'color': 'orange'
    # },
    # 'MINE_entropy': {
    #     'model': Mine(
    #         lr=1e-3, 
    #         batch_size=batch_size, 
    #         patience=int(20), 
    #         iter_num=int(2e+4), 
    #         log_freq=int(100), 
    #         avg_freq=int(10), 
    #         ma_rate=0.01, 
    #         verbose=False,
    #         prefix=prefix_name,
    #         marginal_mode='unif'
    #     ), 
    #     'color': 'purple'
    # }
}

n_samples = batch_size * 20
rhos = [0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999 ]
variables = [2]
widths = list(range(10))

data = {
    'Uniform': {
        'model': Uniform, 
        'kwargs': [
            {
                'n_samples':n_samples, 
                'n_variables':var, 
                'low':0.0, 
                'high': 1.0
            } for var in variables
        ], 
        'varying_param_name': 'n_variables', 
        'x_axis_name': 'numbers of uniform variables', 
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