import scipy.stats as st
import numpy as np
import pdb
import matplotlib.pyplot as pl
import pandas as pd
import json

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
    hist=sig[upp].flatten()
    hist=hist[np.abs(hist)<np.percentile(np.abs(hist),99)]
    
    name=json.dumps({'N':N,'name':'rat','min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})
    pl.hist(hist,density=False,bins='sturges')
    pl.title(name)
    pl.xlabel("value")
    pl.ylabel("Frequency")
    fig.savefig(name+".png")

    return(sig,{'name':'norm_sig','min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})

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

    hist=rat[upp].flatten()
    hist=hist[np.abs(hist)<np.percentile(np.abs(hist),99)]
    
    name=json.dumps({'N':N,'name':'rat','min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})
    pl.hist(hist,density=False,bins='sturges')
    pl.title(name)
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(name+".png")
    
    return(rat,{'name':'rat','min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})