# import model
from model.linear_regression import LinearReg
from model.mine import Mine
from model.kraskov import Kraskov
from data.bimodal import BiModal
from data.gaussian import Gaussian

cpu = 1

batch_size=300


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
    'MINE': {
        'model': Mine(
            lr=1e-4, 
            batch_size=batch_size, 
            patience=int(20), 
            iter_num=int(2e+3), 
            log_freq=int(100), 
            avg_freq=int(10), 
            ma_rate=0.01, 
            verbose=False
        ), 
        'color': 'orange'
    }
}

n_samples = batch_size * 20
rhos = [0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999 ]
widths = list(range(10))

data = {
    'BiModal': {
        'model': BiModal,
        'kwargs': [  # list of params
            {
                'n_samples':n_samples, 
                'mean1':0, 
                'mean2':0, 
                'rho1': rho, 
                'rho2': -rho
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
                'rho': rho
            } for rho in rhos
        ], 
        'varying_param_name': 'rho', 
        'x_axis_name': 'correlation', 
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