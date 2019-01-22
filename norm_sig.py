import scipy.stats as st
import numpy as np
import pdb
import matplotlib.pyplot as pl
import pandas as pd
import json
from math import log

def norm_sig(N,cov):
    sig=np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N)))
    sig=(np.matmul(sig,sig.T)+cov*np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,1)))))
    
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    
    upp=np.triu_indices(N,1)

    fig=pl.figure()
    hist=sig[upp].flatten()
    hist=hist[np.abs(hist)<np.percentile(np.abs(hist),99)].tolist()
    
    name='N:'+str(N)+'|name: diag-'+str(round(log(cov,N),2))
    pl.hist(hist,density=False,bins='sturges')
    pl.title(name)
    pl.xlabel("value")
    pl.ylabel("Frequency")
    fig.savefig(name+".png")

    return(sig,name)

def raw_data(fileName,datName,N):
    data=pd.read_csv(fileName,sep=',').iloc[:,0:N]
    data=np.cov(data,rowvar=False)
    
    diag=np.sqrt(np.diag(data).reshape(-1,1))
    data=(data/np.matmul(np.abs(diag),np.abs(diag).T))
    
    upp=np.triu_indices(N,1)

    hist=data[upp].flatten()
    hist=hist[np.abs(hist)<np.percentile(np.abs(hist),99)].tolist()
    
    name='N:'+str(N)+'|name: '+datName
    pl.hist(hist,density=False,bins='sturges')
    pl.title(name)
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(name+".png")
    
    return(data,name)