import numpy as np
import random
import pandas as pd
import pdb

def raw_data(filename, N):
    data=pd.read_csv(filename)
    data=data.loc[:,data.apply(lambda x: len(np.unique(x)),axis=0)>2]
    sel=random.sample(range(data.shape[1]), N)
    data=data.iloc[:,sel]
    
    cov=np.cov(data,rowvar=False)
    cov=(cov+cov.T)/2

    if N>data.shape[0]:
        U,d,Vt=np.linalg.svd(cov)

        k=np.sum(d>.1)
        D=np.diag(np.append(d[0:k],[d[k-1]]*(N-k)))
        cov=np.matmul(np.matmul(U,D),U.T)
            
    sd=np.diag(1/np.sqrt(np.diag(cov)))
    corr=np.matmul(np.matmul(sd,cov),sd)
    
    return(corr)
    
