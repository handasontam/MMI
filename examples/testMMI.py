import numpy as np
from datetime import datetime
import os

from ..MMI.mmi import *
from ..model.linear_regression import LinearReg
from ..model.logistic_regression import LogisticReg
from ..model.cart_regression import cartReg
from ..data.uniform import Uniform


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot3D(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = X.shape[0]
    ax.scatter(X[:,0], X[:,1], X[:,2])
    plt.show()

def test():
    n_samples = 500
    n_test = 3
    cv_fold = 2

    print(os.getcwd())
    prefix_name = "MMI/Output/{0}/".format(datetime.now())
    os.mkdir(prefix_name)

    for n_var in range(3, 3+n_test):
        #make loop specific folder
        prefix_name_loop = "{0}n_var={1}/".format(prefix_name, n_var)
        os.mkdir(prefix_name_loop)

        mmi = MMI(n_var, prefix=prefix_name_loop)
        X = Uniform(n_samples, n_var)
        data = X.data #n_sample * n_var
        # plot3D(data)
        cond_ent_Reg = cartReg(cv_fold).predict_Cond_Entropy(data)

        # cond_ent_tables = []
        # cond_ent_tables.append(cond_ent_Reg)
        # cond_ent_tables = np.array(cond_ent_tables)

        # subset_ent_tables = mmi.condEntTables2subsetEntTables(cond_ent_tables)
        # ET_Reg = subset_ent_tables[0,:]


        cond_ent_tables = dict()
        cond_ent_tables['Cart_Reg'] = cond_ent_Reg

        subset_ent_tables = mmi.condEntTables2subsetEntTables(cond_ent_tables)
        ET_Reg = subset_ent_tables['Cart_Reg']
        

        #Save result to files
        np.savetxt("{0}cond_ent_Reg.txt".format(prefix_name_loop), cond_ent_Reg)

        gamma = mmi.computeMMI(ET_Reg)
        print("gamma of n_var={0} = {1}".format(n_var, gamma))


if __name__=="__main__":
        plotMMI()
