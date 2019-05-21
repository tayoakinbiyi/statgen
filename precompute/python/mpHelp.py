import numpy as np
import pandas as pd
import pdb
import psutil

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def mpHelp(minMaxK,rhoBar):
    ebb=pd.DataFrame(np.empty([(minMaxK['maxK']-minMaxK['minK']+1).sum(),2]), columns=['pi','N','k','ggnull','gbj'],dtype='float32')

    piList=minMaxK['pi'].drop_duplicates().values.flatten()
    
    piVar=varNoMu(piList,rhoBar)  
    
    count=0
    # ggnull loop
    for piInd in range(len(piList)):
        pi=piList[piInd]
        
        var=piVar[piInd]        
        rho=var/(pi*(1-pi))
        gamma=rho/(1-rho)

        piData=minMaxK.loc[pi]

        piNGood=piData[gamma>=np.maximum(-pi / (piNGood['N']-1),-(1-pi) / (piData['N']-1),axis=1)]
        piNBad=piData[gamma<np.maximum(-pi / (piNGood['N']-1),-(1-pi) / (piData['N']-1),axis=1)]

        prGood=np.empty(piNGood.shape)
        prBad=np.empty(piNBad.shape)
        
        NGoodLen=piNGood.shape[0]
        NBadLen=piNBad.shape[0]
        
        if len(piNGood)>0:
            kLen=(maxK-minK+1)
            maxN=np.max(piNGood['N'])
            maxK=np.max(piNGood['k'])+1
        
            baseOne=np.append([0],np.cumsum(np.log(pi+gamma*np.array(range(0,maxK+1)))))
            baseTwo=np.cumsum(np.log(1-pi+gamma*np.array(range(maxN))))
            baseThree=np.cumsum(np.log(1+gamma*np.array(range(maxN))))
            baseCr=np.log(np.array([scipy.special.comb(row['N'],range(row['maxK']+2)).tolist() for index, row in piNGood.iterrows()]))

            PrGood={row['N']:np.exp(np.log(scipy.special.comb(row['N']))+baseOne[range(row['maxK']+2)]+baseTwo[row['N']-
                np.arange(row['maxK']+2)]-baseThree[row['N']]) for index,row in piNGood.iterrows()}            
            
        if len(piNBad)>0:
            PrBad={row['N']:np.array([np.nan]*(row['maxK']+2)) for index,row in piNBad.iterrows()}
            
        NkOverN=pd.DataFrame([[row['N'],k/row['N']] for index,row in piNGood for k in range(row['minK'], row['maxK']+1)],
            columns=['N','kOverN']).sort_values(by='kOverN').set_index('kOverN')
        
        kOverNN={i:NkOverN.loc[i]['N'].values.flatten() for i in NkOverN.index}
        
        for kOverN in kOverNN:
            var=varWithMu(pi,kOverN.keys(),rhoBar,sf)       
            rho=var/(pi*(1-pi))
            gamma=rho/(1-rho)
        
        
        
        ebb.iloc[count:(count+kLen)]=pd.DataFrame({'pi':pi,'N':nkListGood[:,0],'k':nkListGood[:,1],'ggnull':1-(np.sum(Pr[0:rMin])+
            np.cumsum(Pr[rMin:rMax+1]))},index=range(count,count+rLen),dtype='float32')
        
        else:
            ebb.iloc[count:(count+rLen)]=pd.DataFrame({'binEdge':rLam+np.arange(rMin,rMax+1),'ggnull':np.nan},
                index=range(count,count+rLen),dtype='float32')

        piData=minMaxK.loc[pi]
        piData=piData.reset_index(level=1,drop=True).reset_index()
        piData.insert(piData.shape[1],'kOverN',piData['k']/piData['N'])
        
        for index,row in piData.iterrows():
            N

        kLen=(maxK-minK+1)
        rRho=(rVar-N*rLam*(1-rLam))/(N*(N-1)*rLam*(1-rLam))
        rGamma=rRho/(1-rRho)

        row=row.astype('float32')
        N=row['N']
        pi=row['pi']
        minK=row['minK'].astype(int)
        maxK=row['maxK'].astype(int)
        var=row['var']
        
        
            
        count+=rLen

    return(ebb)