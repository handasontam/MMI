import numpy as np

class GaussianEntropy():
    def __init__(self, CovMat):
        self.Sigma = np.array(CovMat)
        if self.Sigma.shape[0] != self.Sigma.shape[1]:
            raise ValueError("CovMat must be a square matrix")

    def h(self, S):
        c = np.log(2*np.pi*np.e) /2
        L = np.linalg.cholesky(S)
        k = S.shape[0]
        det = k*c
        for i in range(k):
            det += np.log(L[i,i])
        return det

    def __call__(self, B):
        n = len(B)
        S = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                S[i,j] = self.Sigma[B[i], B[j]]
        return self.h(S)

    def predict_Cond_Entropy(self, X):
        from ..MMI.IC.AIC import TableEntropy
        n_var = X.shape[1]
        numCond = 2**(n_var-1)
        cond_ent = np.zeros((n_var, numCond))
        for Resp in range(n_var):
            for sI in range(numCond):
                subset = TableEntropy.subsetVector(n_var - 1, sI)
                subset = np.array(subset)
                cond = []
                for element in subset:
                    if element >= Resp:
                        element += 1
                    cond.append(int(element))
                # self.savefig()
                whole = cond.copy()
                whole.append(Resp)
                cond_ent[Resp, sI] = self(whole) - self(cond)
        return cond_ent
