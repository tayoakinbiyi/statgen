import scipy.stats as st
import numpy as np
import pdb
import matplotlib.pyplot as pl
import pandas as pd

def norm_sig(N,cov):
    sig=np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N)))
    sig=(np.matmul(sig,sig.T)+cov*np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,1)))))
    
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    
    upp=np.triu_indices(N,1)

    avg_cor=np.round(np.mean(sig[upp].tolist()),2)
    max_cor=np.round(max(sig[upp].tolist()),2)
    min_cor=np.round(min(sig[upp].tolist()),2)
   
    fig=pl.figure()
    pl.hist(sig[upp],density=False,bins='sturges')
    pl.title(str({'N':N,'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor}))
    pl.xlabel("value")
    pl.ylabel("Frequency")
    fig.savefig(str({'N':N,'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})+".png")

    return(sig,{'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})

def rat_data(N):
    rat=pd.read_csv('rat.csv',sep='\t').iloc[:,0:N]
    rat=np.cov(rat,rowvar=False)
    
    diag=np.sqrt(np.diag(rat).reshape(-1,1))
    rat=(rat/np.matmul(np.abs(diag),np.abs(diag).T))
    
    upp=np.triu_indices(N,1)
    avg_cor=np.round(np.mean(rat[upp].tolist()),2)
    max_cor=np.round(max(rat[upp].tolist()),2)
    min_cor=np.round(min(rat[upp].tolist()),2)
    pct_neg_cor=np.mean((rat[upp]>0).tolist())

    fig = pl.hist(rat[upp].flatten().tolist(),density=False)
    pl.title('Mean')
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(str({'N':'rat_'+str(N),'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})+".png")
    
    return(rat,{'N':N,'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})