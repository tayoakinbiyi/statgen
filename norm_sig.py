import scipy.stats as st
import numpy as np

def norm_sig(N):
    sig=st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N))
    sig=(np.matmul(sig,sig.T)+np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=20**2,size=(N,1))))).round(2)
    diag=np.diag(sig).reshape(-1,1)
    sig=(sig/np.sqrt(np.matmul(np.abs(diag),np.abs(diag).T))).round(2)
    return(sig)