import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
import time

from ELL.util import *
from multiprocessing import cpu_count
from limix.model.lmm import LMM
from numpy_sugar.linalg import economic_qs
from utility import *

def runH1(mu,n_assoc,wald,Y,K,M,snps,eta):     
    numSnps=snps.shape[1]
    numTraits=Y.shape[1]
                            
    b_wald=bufCreate('waldH1',[numSnps,numTraits])
    b_wald[0][:]=wald
    
    b_eta=bufCreate('etaH1',[numTraits])
    b_eta[0][:]=eta
    
    U,D,Vt=np.linalg.svd(K)

    pids=[]
    numCores=cpu_count()
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        #runH1Help(mu,n_assoc,Y,U,D,M,snps,snpRange,b_wald,b_eta)
        pids+=[remote(runH1Help,mu,n_assoc,Y,U,D,M,snps,snpRange,b_wald,b_eta)]
        
    for pid in pids:
        os.waitpid(0, 0)
    
    return(bufClose(b_wald))

def runH1Help(mu,n_assoc,Y,U,D,M,snps,snpRange,b_wald,b_eta):
    numSubjects,numTraits=Y.shape
    m=M.shape[1]
    
    for snpInd in range(len(snpRange)):
        snp=snpRange[snpInd]
        print('runH1 : {} of {}'.format(snpInd, len(snpRange)),flush=True)
        traitRange=np.random.choice(range(numTraits),n_assoc)
        maf=np.mean(snps[:,snp])/2
        Mhat=np.concatenate([M,snps[:,snp:snp+1]],axis=1)

        for trait in traitRange:
            Linv=U@np.diag((b_eta[0][trait]*D+(1-b_eta[0][trait]))**(-0.5))
            g=Linv@snps[:,snp:snp+1]
            g-=np.mean(g)
            
            y=Linv@(Y[:,trait:trait+1]+(mu/np.sqrt(2*maf*(1-maf)))*np.random.choice([-1,1],size=1)*snps[:,snp:snp+1])
            y-=np.mean(y)
            ans=(g.T@y)/(g.T@g)

            b_wald[0][snp,trait]=ans
            b_wald[1].flush()
    
    return()
