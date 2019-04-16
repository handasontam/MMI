import numpy as np
from datetime import datetime
import os


from ..model.mmi import MMI

def testEntropyTableIndexing(numRV):
    mmi = MMI(numRV)
    numCol = 2**(numRV-1)
    print("print Entropy table layout")
    title = "Col\tRow\tIndex\tSet"
    print(title)
    for i in range(numRV):
        for j in range(numCol):
            cond = np.array(mmi.index2cond(i, j))
            inde = mmi.cond2index(i, cond)
            txt = "{0}\t{1}\t{2}\t{3}".format(i,j,inde,cond)
            print(txt)
    print("end")

if __name__=="__main__":
    testEntropyTableIndexing(6)
