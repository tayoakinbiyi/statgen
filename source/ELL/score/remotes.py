import ray
import numpy as np
import pdb
import os

@ray.remote
def scoreHelp(kRange,pvals,lamEllByK,check):  
    print('scoreHelp ({}): lamEllByK {}'.format(os.getpid(), id(lamEllByK)))

    for k in kRange:        
        check[:,k]=[k,np.mean(pvals[:,k]<lamEllByK[0,k]),np.mean(pvals[:,k]>lamEllByK[-1,k])]
        sortOrd=np.argsort(pvals[:,k])
        pvals[sortOrd,k]=np.clip(np.searchsorted(lamEllByK[:,k],pvals[sortOrd,k],side='left'),0,lamEllByK.shape[0]-1)        

    return()
