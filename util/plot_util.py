import numpy as np


def getHeatMap(ax, xs, ys, z, sampleNum=0):
    """[summary]
    
    Arguments:
        ax {[type]} -- [description]
        xs {[type]} -- [description]
        ys {[type]} -- [description]
        z {function} -- [description]
    
    Keyword Arguments:
        sampleNum {int} -- [description] (default: {0})
    
    Returns:
        [type] -- [description]
    """
    x_column_vector = xs.flatten()[:,None]
    y_column_vector = ys.flatten()[:,None]
    Z = z(x_column_vector, y_column_vector)
    # Z = self.mine_net(torch.FloatTensor(np.hstack((x_column_vector,y_column_vector)))).detach().numpy()
    # Z = [self.mine_net(torch.FloatTensor([[xs[i,j], ys[i,j]]])).item() for j in range(ys.shape[0]) for i in range(xs.shape[1])]
    Z = np.array(Z).reshape(xs.shape[1], ys.shape[0])
    Z = Z[:-1, :-1]
    z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()
    c = ax.pcolormesh(xs, ys, Z, cmap='RdBu', vmin=z_min, vmax=z_max)
    # set the limits of the plot to the limits of the data
    # ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    return ax, c