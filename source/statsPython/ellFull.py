import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb
import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from scipy.stats import norm,beta

def ellFull(parms,z,ellDSet,L):
    numTraits=parms['parms'][2]
    maxD=int(np.max(ellDSet*numTraits))

    pvals=np.sort(2*norm.sf(np.abs(z)))

    ell=np.empty(shape=[int(parms['parms'][-1][-1]),maxD])
    for i in range(maxD):
        ell[:,i]=beta.cdf(pvals[:,i],i+1,numTraits-i)
    ell=np.concatenate([np.min(ell[:,0:int(ellDSet[j]*numTraits)],axis=1).reshape(-1,1) for j in range(len(ellDSet))],axis=1)   

    zRef=np.matmul(norm.rvs(size=[int(parms['refReps']),numTraits]),L.T)

    pvals=np.sort(2*norm.sf(np.abs(zRef)))

    ellRef=np.empty(shape=[int(parms['refReps']),maxD])
    for i in range(maxD):
        ellRef[:,i]=beta.cdf(pvals[:,i],i+1,numTraits-i)
    ellRef=np.concatenate([np.min(ellRef[:,0:int(ellDSet[j]*numTraits)],axis=1).reshape(-1,1) for j in range(len(ellDSet))],axis=1)   

    monteCarlo=np.empty(shape=ell.shape)
    for i in range(len(ellDSet)):
        sortOrd=np.argsort(ell[:,i],axis=0)
        monteCarlo[sortOrd,i]=(1+np.searchsorted(np.sort(ellRef[:,i]),ell[sortOrd,i]))/(len(ellRef)+1)

    return(monteCarlo,ell)
    