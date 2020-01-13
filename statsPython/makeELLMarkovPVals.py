from opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

def makeELLMarkovPVals(parms):
    transOnly=parms['transOnly']
    numCores=parms['numCores']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    
    ellDSet=parms['ellDSet']
    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)
    traitLoc=traitData.loc[traitData['chr'].isin(traitChr),'chr'].values.flatten()

    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')    
    
    if not transOnly:
        N=len(traitLoc)
        ELLInverse=pd.read_csv('ELL/ELLInverse-'+str(N),header=0,index_col=None,sep='\t')
        L=LZCorr

    for ind in range(len(snpChr)):
        snp=snpChr[ind]

        if transOnly:
            N=sum(traitLoc!=snp)
            L=LZCorr[traitLoc!=snp,traitLoc!=snp]
            ELLInverse=pd.read_csv('ELL/ELLInverse-'+str(N),header=0,index_col=None,sep='\t')
           
        for dParm in ellDSet:
                
            d=int(N*dParm)
                            
            corr=np.matmul(L,L.T)
            offDiag=corr[np.triu_indices(N,1)].flatten()
            offDiagVec=ro.FloatVector(tuple(offDiag))

            stat=np.loadtxt('stats/'+str(dParm)+'-'+str(snp),delimiter='\t')
            statOrd=np.argsort(stat)
            loc=np.minimum(len(ELLInverse)-1,ELLInverse['ell'].searchsorted(stat[statOrd]))
            
            inverse=ELLInverse.iloc[loc,range(1,d+1)].values[np.argsort(statOrd),:]

            segLen=int(np.ceil(len(stat)/numCores))
            futures=[]
            with ProcessPoolExecutor(numCores) as executor: 
                for core in range(numCores):
                    statRange=np.arange(core*segLen,min((core+1)*segLen,len(stat)))
                    #makeELLMarkovPValsHelp(inverse[statRange],offDiagVec,core,d,N)
                    futures+=[executor.submit(makeELLMarkovPValsHelp,inverse[statRange],offDiagVec,core,d,N)]

                df=[]
                for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                    df+=[f.result()]
                
            pval=pd.concat(df,axis=0).sort_values(by=['core','ind'])['pval'].values.flatten()
                
            DBLog(str(dParm)+'-'+str(snp)+' stat ['+str(len(stat))+','+str(np.min(stat))+
                  ','+str(np.mean(stat))+','+str(np.max(stat))+'] pval ['+str(len(pval))+
                  ','+str(np.min(pval))+','+str(np.mean(pval))+','+str(np.max(pval))+']',parms)

            np.savetxt('pvals/ell_'+str(dParm)+'_Markov-'+str(snp),pval,delimiter='\t')

            print('finished '+str(dParm)+' snp '+str(snp),flush=True)
        
    return()

def makeELLMarkovPValsHelp(inverse,offDiag,core,d,N):    
    gbj=importr('GBJ')
    
    df=pd.DataFrame(index=range(len(inverse)),columns=['core','ind','pval'])
    for ind in range(len(inverse)):
        row=inverse[ind]
        bounds=ro.FloatVector(np.array(row[0:d].tolist()+[row[d-1]]*(N-d))[::-1])
        df.iloc[ind,:]=[core,ind,gbj.ebb_crossprob_cor_R(d=N, bounds=bounds, correlations=offDiag)[0]]
    
    return(df)