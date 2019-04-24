import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def save_train_curve(train_loss, valid_loss, figName):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

def varEntropy(y):
    return np.log(np.var(y)*np.pi*2)/2

def mseEntropy(clf, X, y):
    y_est = clf.predict(X)
    return np.log(mean_squared_error(y, y_est)*np.pi*2)/2

def unifEntropy(y, high=1.0, low=0.0):
    return np.log(high-low)

def ShannonEntropy(y):
    from .model import ifestimators as ife
    params = ife.Struct()
    funPara = ife.Struct()
    return ife.octave.shannonEntropy(y, funPara, params)
