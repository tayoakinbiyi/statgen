import numpy as np
import pdb
import os

def scoreHelp(kRange,b_pvals,lamEllByK,b_check):      
    for k in kRange:        
        b_check[0][:,k]=[k,np.mean(b_pvals[0][:,k]<lamEllByK[0,k]),np.mean(b_pvals[0][:,k]>lamEllByK[-1,k])]
        sortOrd=np.argsort(b_pvals[0][:,k])
        b_pvals[0][sortOrd,k]=np.clip(np.searchsorted(lamEllByK[:,k],b_pvals[0][sortOrd,k],side='left'),0,lamEllByK.shape[0]-1)        
    
    return()
