import torch

def update_moving_average(ma, x, ma_rate):
    """[update the moving average ma when observing new value x]
    
    Arguments:
        ma {[type]} -- [description]
        x {[type]} -- [description]
        ma_rate {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return (1-ma_rate)*ma + ma_rate*torch.mean(x)