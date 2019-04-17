import numpy as np
import pandas as pd
import pdb
import psutil
from scipy.stats import norm, beta
from python.qq_var import *

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def mpHelp(minMaxK,normSF,N,rhoBar,pairwise_cors):
    cr=newC(N)
    
    ebb=pd.DataFrame(np.empty([(minMaxK['maxK']-minMaxK['minK']+1).sum(),3]), columns=['binEdge','ggnull','gbj'],dtype='float32')
    
    normSFLoc=normSF.z.searchsorted(np.abs(norm.ppf(minMaxK['binEdge']/2)))
    sf=normSF.sf.values.flatten()
    z=normSF.z.values.flatten()
    B=len(sf)
    
    minMaxK=minMaxK.reset_index(drop=True)

    count=0
    for index, row in minMaxK.iterrows():
        row=row.astype('float32')
        
        binEdge=row['binEdge']
        minK=row['minK'].astype(int)
        maxK=row['maxK'].astype(int)
        
        varNoMu=row['varNoMu']

        lamNoMu=binEdge
        kLen=(maxK-minK+1)
        rhoNoMu=(varNoMu-N*binEdge*(1-binEdge))/(N*(N-1)*binEdge*(1-binEdge))
        gammaNoMu=rhoNoMu/(1-rhoNoMu)
        
        # gbj
        gbj=np.array([np.nan]*kLen)

        loc=normSFLoc[index]
        muList=sf[:loc+1][::-1]+np.append(sf[loc:],[0]*max(0,2*loc-B+1))[:loc+1]
        muLoc=np.searchsorted(muList,(np.arange(minK,maxK+1)+1)/N)
        mu=z[loc]-z[loc-muLoc]
        
        varWithMu= getVarWithMu(z[loc],N,mu,rhoBar)
        lamWithMu=(np.arange(minK,maxK+1)+1)/N
        rhoWithMu=(varWithMu-N*lamWithMu*(1-lamWithMu))/(N*(N-1)*lamWithMu*(1-lamWithMu))
        gammaWithMu=rhoWithMu/(1-rhoWithMu)

        if gammaNoMu>=max(-lamNoMu / (N-1),-(1-lamNoMu) / (N-1)):   
            baseOne=np.cumsum(np.append([0],np.log(lamNoMu+gammaNoMu*np.arange(maxK+1))))
            baseTwo=np.cumsum(np.log(1-lamNoMu+gammaNoMu*np.array(range(N))))[-(maxK+2):][::-1]
            baseThree=np.sum(np.log(1+gammaNoMu*np.array(range(N))))
            baseCr=cr[0:(int(maxK)+2)]
            lPr=baseCr+baseOne+baseTwo-baseThree
            Pr=np.exp(lPr) # goes from 0 to maxK inclusive
            
            ggnull=1-(np.sum(Pr[0:minK])+np.cumsum(Pr[minK:maxK+1]))
            
            pdb.set_trace()
            gbjK=np.arange(minK,maxK+1)[(binEdge<(np.arange(minK,maxK+1)+1)/N)&(gammaWithMu>=np.maximum(-lamWithMu/(N-1),
                -(1-lamWithMu)/(N-1)))]
            for i_k in range(kLen):
                baseOne=np.sum(np.log(lamWithMu[i_k]+gammaWithMu[i_k]*np.arange(gbjK[i_k]+1)))
                baseTwo=np.sum(np.log(1-lamWithMu[i_k]+gammaWithMu[i_k]*np.arange(N-gbjK[i_k]-1)))
                baseThree=np.sum(np.log(1+gammaWithMu[i_k]*np.arange(N)))
                baseCr=cr[int(gbjK[i_k])+1]
                gbj[i_k]=(baseCr+baseOne+baseTwo-baseThree)-lPr[gbjK[i_k]+1]
                
            ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':binEdge+np.arange(rMin,rMax+1),'ggnull':ggnull,'gbj':gbj},
                index=range(count,count+rLen),dtype='float32')
        else:
            ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':binEdge+np.arange(rMin,rMax+1),'ggnull':np.nan,'gbj':np.nan},
                index=range(count,count+rLen),dtype='float32')
            
        count+=rLen

    return(ebb)