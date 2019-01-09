import scipy.stats as st
import numpy as np

def norm_sig(N,cov):
    sig=st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N))
    sig=(np.matmul(sig,sig.T)+np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=cov**2,size=(N,1)))))
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    return(sig)