from oct2py import octave, Oct2Py, Struct
from os import getcwd
import numpy as np

cwd = getcwd()
path2estimators = "{0}/model/ifestimators/estimators/".format(cwd)
octave.addpath(path2estimators)
path2kde = "{0}/model/ifestimators/kde/".format(cwd)
octave.addpath(path2kde)
# octave.ifSetup(cwd)
octave.warning('off', 'Octave:possible-matlab-short-circuit-operator')

# octave.eval('demo5()')

X = np.random.normal(size=100)
# oc = Oct2Py()
params = Struct()
funPara = Struct()
Y = octave.shannonEntropy(X, params, funPara)
print(Y)
