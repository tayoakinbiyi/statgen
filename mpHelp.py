import numpy as np
import pandas as pd
import pdb
import psutil

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def mpHelp(minMaxK,N):
    cr=newC(N)
    #pdb.set_trace()

    ebb=pd.DataFrame(columns=['binEdge','ebb'],dtype='float32')

    for index, row in minMaxK.iterrows():
        row=row.astype('float32')
        rLam=row['binEdge']
        rMin=row['minK'].astype(int)
        rMax=row['maxK'].astype(int)
        rVar=row['var']

        rLen=(rMax-rMin+1)
        rRho=(rVar-N*rLam*(1-rLam))/(N*(N-1)*rLam*(1-rLam))
        rGamma=rRho/(1-rRho)

        if rGamma>=max(-rLam / (N-1),-(1-rLam) / (N-1)):   
            baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.array(range(0,rMax)))))
            baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.array(range(N))))[-(rMax+1):][::-1]
            baseThree=np.sum(np.log(1+rGamma*np.array(range(N))))
            baseCr=cr[0:(int(rMax)+1)]

            Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
            ebb=ebb.append(pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ebb':1-(np.sum(Pr[0:rMin])+np.cumsum(Pr[rMin:rMax+1]))},
                dtype='float32'))
        else:
            ebb=ebb.append(pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ebb':np.nan},dtype='float32'))

    return(ebb)