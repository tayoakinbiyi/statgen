import scipy.stats as st
import numpy as np
import pdb

def norm_sig(N,cov):
<<<<<<< HEAD
    sig=np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N)))
    sig=(np.matmul(sig,sig.T)+cov*np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,1)))))
    
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    return(sig)

def rat_data(N):
    rat=pd.read_csv('rat.csv',sep='\t')[:,0:N]
    return(np.cov(rat,rowvar=False))
=======
    sig=st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N))
    sig=(np.matmul(sig,sig.T)+np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=cov**2,size=(N,1)))))
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    return(sig)
>>>>>>> 7c9b9e8f4e929989108941dd13bb803a12819d35
