from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import norm,beta
import psutil
import pdb

def ghc(numSegs,parms):
    cpus=parms['cpus']
    Reps,N=z.shape
    delta=parms['delta']
    
    futures=[]
    ghc=pd.DataFrame(dtype='float32')
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(numSegs):
            futures.append(executor.submit(ghcHelp,parms))

    for f in as_completed(futures):
        ghc=ghc.append(f.result())
    
    return(ghc)

def ghcHelp(parms):
    name=parms['name']
    local=parms['local']
    path=local+name
        
    subprocess.run(['Rscript',path+'R/myGHC.R',str(i),path])
    
    result=pd.read_csv(path+'ghc/ghc-'+str(i)+'.csv')
    ghc=pd.DataFrame({'Type':'ghc','Value':result['pval']})
    
    return(ghc)

