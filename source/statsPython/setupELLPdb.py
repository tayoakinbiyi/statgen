import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from scipy.stats import norm, beta, binom
import pdb
import sys

from opPython.DB import *

def setupELL(parms):
    transOnly=parms['transOnly']
    traitChr=parms['traitChr']
    pdb.set_trace()
    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')   
    
    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)
    traitLoc=traitData['chr'][traitData['chr'].isin(traitChr)].values.flatten()
    
    if transOnly:
        for snp in snpChr:
            setupELLSnp(LZCorr[traitLoc!=snp,traitLoc!=snp],parms)
    else:
        setupELLSnp(LZCorr,parms)

    return()

def setupELLSnp(L,parms):
    ellDSet=parms['ellDSet']
    ellBetaPpfEps=parms['ellBetaPpfEps']    
    ellKRanLowEps=parms['ellKRanLowEps']
    ellKRanHighEps=parms['ellKRanHighEps']
    ellKRanNMult=parms['ellKRanNMult']
    binsPerIndex=parms['binsPerIndex']
    numCores=parms['numCores']
    minELLDecForInverse=parms['minELLDecForInverse']

    N=L.shape[0]     
    print('setupELLSnp '+str(N),flush=True)
    
    dParm=max(ellDSet)
    d=int(N*dParm)
    offDiag=np.matmul(L,L.T)[np.triu_indices(N,1)].flatten()
      
    bins=np.linspace(0,1,int(N*binsPerIndex)+1)
    bins=np.append(np.linspace(0,bins[1],binsPerIndex),bins[2:-1])

    kRan=np.arange(1,d+1)
    kRanDown=np.maximum(1,np.minimum(kRan-ellKRanNMult*N,kRan*ellKRanLowEps)).astype(int)
    kRanUp=np.minimum(N,np.maximum(kRan+ellKRanNMult*N,kRan*ellKRanHighEps)).astype(int)

    minLam=beta.ppf(ellBetaPpfEps,kRanDown,N+1-kRanDown)
    maxLam=beta.ppf(1-ellBetaPpfEps,kRanUp,N+1-kRanUp)
    minMaxLamPerK=pd.DataFrame({'minLam':minLam,'maxLam':maxLam})
      
    minB=np.digitize(minMaxLamPerK['minLam'],bins).astype(int)-1   
    maxB=np.digitize(minMaxLamPerK['maxLam'],bins).astype(int)-1
    maxB[0]=maxB[1]
    
    bins=bins[1:max(maxB)+2]
    numBins=len(bins)
     
    minMaxKPerBin=pd.DataFrame({'binEdge':bins,'z':-norm.ppf(bins/2)})
    minK=pd.DataFrame({'maxB':maxB,'k':range(d)}).groupby(by='maxB').apply(lambda df: pd.DataFrame({'maxB':df.maxB.iloc[0],
        'k':df.k.min()},index=[0])).reset_index(drop=True)
    minMaxKPerBin.insert(1,'minK',minK.k.iloc[minK.maxB.searchsorted(range(numBins),side='left')].values)

    maxK=pd.DataFrame({'minB':minB,'k':range(d)}).groupby(by='minB').apply(lambda df: pd.DataFrame({'minB':df.minB.iloc[0],
        'k':df.k.max()},index=[0])).reset_index(drop=True)
    minMaxKPerBin.insert(2,'maxK',maxK.k.iloc[maxK.minB.iloc[1:].searchsorted(range(numBins),side='right')].values)
            
    end=np.cumsum(maxB-minB+1) 
    start=np.append([0],end[:-1])
    
    minMaxBinPerK=pd.DataFrame({'minBin':minB,'maxBin':maxB})
    startStopBinsPerK=pd.DataFrame({'start':start,'end':end},dtype='int')
    
    rhoBar=getRhoBar(offDiag)
    gamma=[]
    
    print('getGamma',flush=True)
    futures=[]
    with ProcessPoolExecutor(numCores) as executor:    
        for core in range(numCores):
            binRange=np.arange(core*int(np.ceil(numBins/numCores)),min((core+1)*int(np.ceil(numBins/numCores)),numBins))
            if len(binRange)==0:
                continue

            futures+=[executor.submit(getGamma,core,bins[binRange],rhoBar)]
        
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            gamma+=[f.result()]
    
    gamma=pd.concat(gamma,axis=0).sort_values(by='binEdge').reset_index(drop=True)   
    minMaxKPerBin.insert(minMaxKPerBin.shape[1],'gamma',gamma['gamma']) # binEdge,z,min,max, gamma
    
    print('setupELLHelp',flush=True)
    futures=[]
    ebb=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for core in range(numCores):
            binRange=np.arange(core*int(np.ceil(numBins/numCores)),min((core+1)*int(np.ceil(numBins/numCores)),numBins))
            if len(binRange)==0:
                continue
            futures+=[executor.submit(setupELLHelp,N,core,minMaxKPerBin.iloc[binRange])]
                                      
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
             ebb+=[f.result()]   
                
    ebb=pd.concat(ebb,axis=0).sort_values(by=['k','binEdge']).reset_index(drop=True)
    
    minMaxEllPerK=ebb.groupby('k')['ell'].apply(lambda df: pd.DataFrame({'min':df.min(),'max':df.max()},index=[0])
        ).reset_index().drop(columns='level_1')
    
    print('min Ell '+str(minMaxEllPerK['min'].min())+' max Ell '+str(minMaxEllPerK['max'].max()))
    
    minMaxEllPerK=minMaxEllPerK[(minMaxEllPerK['min']>10**(-minELLDecForInverse))|
                                (minMaxEllPerK['max']<1-10**(-minELLDecForInverse))]
    if len(minMaxEllPerK)>0:
        print(minMaxEllPerK,flush=True)
        sys.exit()
        
    ELLVals=np.arange(1,10**minELLDecForInverse)/10**minELLDecForInverse
    ELLInverse=pd.DataFrame(columns=range(d))
    ELLInverse.insert(0,'ell',ELLVals)
    
    print('setupELLInverseHelp',flush=True)
    futures=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for core in range(numCores):
            kRange=np.arange(core*int(np.ceil(d/numCores)),min((core+1)*int(np.ceil(d/numCores)),d))
            if len(kRange)==0:
                continue
            
            ebb.loc[ebb['k'].isin(kRange)].to_csv('ELL/ebb-'+str(core),index=False,sep='\t')
            #setupELLInverseHelp(ebb.loc[ebb['k'].isin(kRange),['binEdge','ell','k']],kRange,ELLVals,N,d,parms)
            futures+=[executor.submit(setupELLInverseHelp,core,kRange,ELLVals,N,d,parms)]
            
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            ans=f.result()
            ELLInverse.loc[:,ans[0]]=ans[1]
    
    ELLInverse.to_csv('ELL/ELLInverse-'+str(N),index=False,sep='\t')  
    print('finished setupELL '+str(N),flush=True)
    
    return()

def setupELLInverseHelp(core,kRange,ELLVals,N,d,parms): 
    minELLDecForInverse=parms['minELLDecForInverse']
    
    ebb=pd.read_csv('ELL/ebb-'+str(core),index_col=None,header=0,sep='\t')
    
    minELLForInverse=10**(-minELLDecForInverse)
    maxELLForInverse=1-10**(-minELLDecForInverse)
    
    ELLInverse=np.empty([len(ELLVals),len(kRange)])
    for kInd in range(len(kRange)):
        k=kRange[kInd]
        ELL=ebb.loc[ebb['k']==k]
        
        t_ELL=ELL.sort_values(by='ell')
        ELLInverse[:,kInd]=t_ELL['z'].iloc[np.minimum(len(t_ELL)-1,t_ELL['ell'].searchsorted(ELLVals))]
        
        ELL[['binEdge','ell']].to_csv('ELL/ELL-'+str(N)+'-'+str(k),index=False,sep='\t')
        
        print('setupELL '+str(N)+' '+str(k+1)+' of '+str(d),flush=True)
                
    return((kRange,ELLInverse))
    
def getRhoBar(pairwise_cors):
    return([np.mean(pairwise_cors), np.mean(pairwise_cors**2), np.mean(pairwise_cors**3), np.mean(pairwise_cors**4),
        np.mean(pairwise_cors**5), np.mean(pairwise_cors**6), np.mean(pairwise_cors**7), np.mean(pairwise_cors**8),
        np.mean(pairwise_cors**9), np.mean(pairwise_cors**10)])
    
def getGamma(core,binEdge,rho):
    z=np.abs(norm.ppf(binEdge/2))
    odds=getHerm(z,rho)  
    x=4*norm.pdf(z)**2*odds  
    gamma=x/(binEdge*(1-binEdge)-x)

    return(pd.DataFrame({'binEdge':binEdge,'gamma':gamma}))

def getHerm(z,rho):
    He1 = z**2
    He3 = (z**3-3*z)**2
    He5 = (z**5-10*z**3+15*z)**2
    He7 = (z**7-21*z**5+105*z**3-105*z)**2
    He9 = (z**9-36*z**7+378*z**5-1260*z**3+945*z)**2
    
    odds = ( He1*rho[1]/2 + He3*rho[3]/24 + He5*rho[5]/720 + He7*rho[7]/40320 + He9*rho[9]/3628800 )
    
    return(odds)

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def setupELLHelp(N,core,minMaxKPerBin):
    cr=newC(N)

    ebb=[]
    
    for index, row in minMaxKPerBin.iterrows():
        rLam=row['binEdge']
        rZ=row['z']
        rMin=row['minK'].astype(int)
        rMax=row['maxK'].astype(int)
        rGamma=row['gamma']

        baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.array(range(0,rMax)))))
        baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.array(range(N))))[-(rMax+1):][::-1]
        baseThree=np.sum(np.log(1+rGamma*np.array(range(N))))
        baseCr=cr[0:(int(rMax)+1)]

        Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
        ell=np.maximum(1-np.cumsum(Pr)[rMin:rMax+1],0)
        ebb+=[pd.DataFrame({'binEdge':rLam,'z':rZ,'k':np.arange(rMin,rMax+1),'ell':ell}).reset_index(drop=True)]

    return(pd.concat(ebb,axis=0))
    
    return()
