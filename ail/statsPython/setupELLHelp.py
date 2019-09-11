import numpy as np
import pandas as pd
import pdb
import psutil

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def setupELLHelp(minMaxK,N):
    cr=newC(N)

    ebb=pd.DataFrame(np.empty([(minMaxK['maxK']-minMaxK['minK']+1).sum(),2]), columns=['binEdge','ggnull'],dtype='float32')

    count=0
    for index, row in minMaxK.iterrows():
        row=row.astype('float32')
        rLam=row['binEdge']
        rMin=row['minK'].astype(int)
        rMax=row['maxK'].astype(int)
        rVar=row['var']

        rLen=(rMax-rMin+1)
        rRho=(rVar-N*rLam*(1-rLam))/(N*(N-1)*rLam*(1-rLam))
        rGamma=rRho/(1-rRho)
        
        baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.array(range(0,rMax)))))
        baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.array(range(N))))[-(rMax+1):][::-1]
        baseThree=np.sum(np.log(1+rGamma*np.array(range(N))))
        baseCr=cr[0:(int(rMax)+1)]

        Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
        ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ell':1-(np.sum(Pr[0:rMin])+
            np.cumsum(Pr[rMin:rMax+1]))},index=range(count,count+rLen),dtype='float32')
            
        count+=rLen

    return(ebb)