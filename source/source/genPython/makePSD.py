import numpy as np
import pdb

def makePSD(df,corr=True):
    if corr:
        diag=np.diag(1/np.sqrt(np.diag(df)))
        df=np.matmul(np.matmul(diag,df),diag)
    
    U,D,Vt=np.linalg.svd(df)
    
    if np.min(D)<0:
        D-=np.min(D)
    
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    return(L)