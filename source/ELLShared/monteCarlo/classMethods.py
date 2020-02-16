import ray
import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb

def monteCarlo(self,ref,ell):  
    assert ref.shape[1]==ell.shape[1]
    
    memory('monteCarlo')
    
    monteCarlo=np.zeros(ell.shape)

    for dInd in range(ell.shape[1]):
        sortOrd=np.argsort(ell[:,dInd],axis=0)
        monteCarlo[sortOrd,dInd]=(1+np.searchsorted(np.sort(ref[:,dInd]),ell[sortOrd,dInd]))/(len(ref)+1)
        
    memory('monteCarlo')

    return(monteCarlo)
