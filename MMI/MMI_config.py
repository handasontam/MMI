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
from ..data.uniform import Uniform

cpu = 1

batch_size=300


n_samples = 500
n_test = 3
cv_fold = 2

prefix_name = "MMI/Output/{0}/".format(datetime.now())

# ground truth is plotted in red
model = {
    'Linear Regression': {  # model name, for plotting the legend
        'model': LinearReg(  # initialize the object
            cvFold=3
        ), 
        'color': 'blue'  # for plotting
    }, 
    # 'Kraskov': {
    #     'model': Kraskov(
    #         discrete_features='auto', 
    #         n_neighbors=3, 
    #         random_state=None
    #     ), 
    #     'color': 'green'
    # }, 
    'MINE': {
        'model': Mine(
            lr=1e-4, 
            batch_size=batch_size, 
            patience=int(20), 
            iter_num=int(2e+3), 
            log_freq=int(100), 
            avg_freq=int(10), 
            ma_rate=0.01, 
            verbose=False,
            prefix=prefix_name
        ), 
        'color': 'orange'
    }
}

n_samples = batch_size * 20
rhos = [0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999 ]
variables = [3, 4, 5, 6, 7]
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