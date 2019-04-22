# from distutils.core import setup
# from distutils.extension import Extension

# ext_instance = Extension(
#     'AIC',
#     sources=['pylib.cpp'],
#     libraries=['boost_python-mt']
# )

# setup(
#     name='AIC',
#     version='0.1',
#     ext_modules=[ext_instance]
# )

import numpy as np
import AIC

class GaussianEntropy(AIC.SF):
    def __init__(self, S):
        # AIC.SF.__init__(self)
        self.Sigma = np.array(S)
        if self.Sigma.shape[0] != self.Sigma.shape[1]:
            raise ValueError

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

    def size(self):
        return self.Sigma.shape[0]


if __name__=="__main__":
    k = 2
    sigma = 1
    n = 3

    A = np.zeros([n,n+k])
    for i in range(A.shape[0]):
        A[i, i%k] = 1
        A[i, i+k] = sigma

    S = np.matmul(A, np.transpose(A))

    gsf = GaussianEntropy(S)

    first_node = []
    second_node = []
    gamma = []
    for i in range(S.shape[0]):
        for j in range(i):
            first_node.append(i)
            second_node.append(j)
            I = gsf([i]) + gsf([j]) - gsf([i,j])
            gamma.append(I)

    cl = AIC.CL(n, first_node, second_node, gamma)

    cl_gamma = cl.getCriticalValues()

    cl_gamma = np.array(cl_gamma)
    for gamma_ in cl_gamma:
        P = cl.getPartition(gamma_)
        print(P)


    TE = []
    num_subset = 2**n
    for i in range(num_subset):
        TE.append(gsf(AIC.TableEntropy.subsetVector(n, i)))
    
    print ("TE={0}".format(TE))
    # psp = AIC.AIC(gsf)
    psp = AIC.AIC_TE(TE, n)

    while psp.agglomerate(1e-8, 1e-10):
        P = psp.getPartition(-1)
        CriticalValues = psp.getCriticalValues()
        P_np = np.array(P)
        print (P_np)
        CV = np.array(CriticalValues)
        print (CV)

    psp_gamma = psp.getCriticalValues()
    psp_gamma = np.array(psp_gamma)
    for gamma in psp_gamma:
        P = psp.getPartition(gamma)
        P_np = np.array(P)
        print ("partition at threshold {0} : {1}".format(gamma, P_np))
                