import numpy as np
import pdb
import os

def scoreHelp(kRange,b_pvals,lamEllByK,b_check):      
    for kInd in range(len(kRange)):
        k=kRange[kInd]
        b_check[0][:,k]=[k,np.mean(b_pvals[0][:,kInd]<lamEllByK[0,k]),np.mean(b_pvals[0][:,kInd]>lamEllByK[-1,k])]
        sortOrd=np.argsort(b_pvals[0][:,kInd])
        b_pvals[0][sortOrd,kInd]=np.clip(np.searchsorted(lamEllByK[:,k],b_pvals[0][sortOrd,kInd],side='left'),0,
            lamEllByK.shape[0]-1)        
    
    return()
