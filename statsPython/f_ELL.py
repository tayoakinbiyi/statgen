import pandas as pd
import numpy as np
import pdb
from ail.opPython.DB import *
import sys
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def f_ELL(pval,d,N,parms):
    ellDSet=parms['ellDSet']
    numCores=parms['numCores']
    #pdb.set_trace()
    
    Reps=pval.shape[0]
    
    segLen=int(np.ceil(d/numCores))
    futures=[]
    stats=np.empty([Reps,d])
    with ProcessPoolExecutor(numCores) as executor: 
        for core in range(numCores):
            kRange=np.arange(core*segLen,min((core+1)*segLen,d))
            if len(kRange)==0:
                continue
            #f_ELL_help(pval[:,kRange],kRange,N,parms)
            futures+=[executor.submit(f_ELL_help,pval[:,kRange],kRange,N,parms)]

        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            df,kRange=f.result()
            stats[:,kRange]=df
    
    return(stats)

def f_ELL_help(pval,kRange,N,parms):
    name=parms['name']
    logName=parms['logName']
    #pdb.set_trace()
        
    indexBelow=0
    indexAbove=0
    pctBelow=0
    pctAbove=0
    
    stats=np.empty(pval.shape)
    for kInd in range(len(kRange)):
        k=kRange[kInd]
        ELL=pd.read_csv('ELL/ELL-'+str(N)+'-'+str(k),index_col=None,sep='\t')
        
        sortOrd=np.argsort(pval[:,kInd])
        loc=ELL['binEdge'].searchsorted(pval[sortOrd,kInd]).flatten()

        below=sum(pval[:,kInd]<ELL['binEdge'].min())
        above=sum(pval[:,kInd]>ELL['binEdge'].max())
        
        pctBelow+=below
        pctAbove+=above
        indexBelow+=1.0*(below>0)
        indexAbove+=1.0*(above>0)

        stats[:,kInd]=ELL['ell'].iloc[np.minimum(loc,len(ELL)-1)].iloc[np.argsort(sortOrd)].values.flatten()

    if pctBelow+pctAbove>0:
        pctDenom=pval.shape[0]*pval.shape[1]
        indexDenom=pval.shape[1]
        sys.exit('pctBelow '+str(pctBelow/pctDenom)+' pctAbove '+str(pctAbove/pctDenom)+' indexBelow '+str(indexBelow/indexDenom)+
              ' indexAbove '+str(indexAbove/indexDenom))
                       
    return(stats,kRange)