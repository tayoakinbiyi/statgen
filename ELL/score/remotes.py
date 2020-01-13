import ray
import numpy as np

@ray.remote
def scoreHelp(kRange,pvals,lamEllByK,grid,check):        
    for k in kRange:        
        sortOrd=np.argsort(pvals[:,k])
        pvals[sortOrd,k]=np.searchsorted(lamEllByK[:,k],pvals[sortOrd,k],side='left')
        
        check[:,k]=[k,np.mean(pval[:,k]<lamEllByK[0,k]),np.mean(pval[:,k]>lamEllByK[-1,k])]

    return()
