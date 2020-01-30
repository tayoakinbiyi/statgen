import numpy as np

def makePSD(df):
    U,D,Vt=np.linalg.svd(df)
    
    if np.min(D)<0:
        D-=np.min(D)
    
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    return(L)