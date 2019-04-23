#Ported by Michael Lim
#Original R code is in https://github.com/XianliZeng/JMI
#Original paper:
#X. Zeng, Y. Xia, and H. Tong, “Jackknife approach to the estimation of mutual information,” 
#    Proceedings of the National Academy of Sciences, vol. 115, no. 40, pp. 9956–9961, 2018.


#R to Python interfacing
import rpy2.robjects as robj
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
    
#import jackknife MI estimator library
#requires R 3.4.4 and the jackknife library in the computer
mJMI = rpackages.importr('mJMI')

class Jackknife():
    def __init__(self, n_sim):
        self.n_sim = n_sim
    def predict(self, X):
        xs = X[:, 0]
        ys = X[:, 1]
        
        xs = xs.reshape(-1,1)
        ys = ys.reshape(-1,1)

        #convert data to R matrix
        xr, xc = xs.shape
        x = robj.r.matrix(xs, nrow=xr, ncol=xc)
        yr, yc = ys.shape
        y = robj.r.matrix(ys, nrow=yr, ncol=yc)
  
        #run the jackknife algorithm and return the result in function
        jmi = mJMI.mJMICpp(x, y, self.n_sim)
        return jmi.rx2('mi')[0]