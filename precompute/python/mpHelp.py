import numpy as np
import pandas as pd
import pdb
import psutil

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def mpHelp(minMaxK,normSF,N,gbjBinLen):
    cr=newC(N)

    ebb=pd.DataFrame(np.empty([(minMaxK['maxK']-minMaxK['minK']+1).sum(),3]), columns=['binEdge','ggnull','gbj'],dtype='float32')
    
    normSFLoc=normSF.z.searchsorted(norm.ppf(minMaxK['binEdge'])).values
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
        kLen=(rMaxK-rMinK+1)
        rhoNoMu=(varNoMu-N*binEdge*(1-binEdge))/(N*(N-1)*binEdge*(1-binEdge))
        gammaNoMu=rRhoNoMu/(1-rhoNoMu)
        
        # gbj
        
        gbj=np.array([0]*kLen)

        loc=normSFLoc[index]
        maxDelta=min(loc-1,B-loc-1)
        muList=sf[0:loc+1][::-1]+np.append(sf[loc:],[0]*max(0,2*loc-B+1))
        muLoc=np.searchsorted(muList,np.arange(minK,maxK+1)/N)
        mu=z[loc+muLoc]-z[loc]
        
        varWithMu= getVarWithMu(z[loc],N,mu,rho)
        lamWithMu=(np.arange(minK,maxK+1)+1)/N
        rhoWithMu=(varWithMu-N*lamWithMu*(1-lamWithMu))/(N*(N-1)*lamWithMu*(1-lamWithMu))
        gammaWithMu=rhoWithMu/(1-rhoWithMu)

        if gammaNoMu>=max(-lamNoMu / (N-1),-(1-lamNoMu) / (N-1)):   
            baseOne=np.cumsum(np.append([0],np.log(lamNoMu+gammaNoMu*np.arange(maxK+1))))
            baseTwo=np.cumsum(np.log(1-lamNoMu+gammaNoMu*np.array(range(N))))[-(maxK+2):][::-1]
            baseThree=np.sum(np.log(1+gammaNoMu*np.array(range(N))))
            baseCr=cr[0:(int(maxK)+2)]
            Pr=np.exp(baseCr+baseOne+baseTwo-baseThree) # goes from 0 to maxK inclusive
            
            ggnull=1-(np.sum(Pr[0:minK])+np.cumsum(Pr[minK:maxK+1]))
            
            gbjK=np.arange(kLen)[(binEdge<(np.arange(minK,maxK+1)+1)/N)&(gammaWithMu>=np.maximum(-lamWithMu/(N-1),-(1-lamWithMu)/(N-1)))]
            for i_k in gbjK:
                baseOne=np.sum(np.log(lamWithMu[i_k]+gammaWithMu[i_k]*np.arange(gbjK[i_k]+1)))
                baseTwo=np.sum(np.log(1-lamWithMu[i_k]+gammaWithMu[i_k]*np.arange(N-gbjK[i_k]-1)))
                baseThree=np.sum(np.log(1+gammaWithMu[i_k]*np.arange(N)))
                baseCr=cr[0:(int(gbjK[i_k])+2)]
                gbj[i_k]=np.exp(baseCr+baseOne+baseTwo-baseThree)-Pr[gbjK[i_k]+1]
                
            ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ggnull':ggnull,'gbj':gbj},
                index=range(count,count+rLen),dtype='float32')
        else:
            ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ggnull':np.nan,'gbj':0},
                index=range(count,count+rLen),dtype='float32')
            
        count+=rLen

    return(ebb)