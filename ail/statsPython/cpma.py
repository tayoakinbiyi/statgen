from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil
import subprocess
import os

def cpma(numSegs,parms):       
    cpus=parms['cpus']
    name=parms['name']
    Reps,d=z.shape

    futures=[]
    cpma=pd.DataFrame(dtype='float32')
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(numSegs):
            futures.append(executor.submit(cpmaHelp,i,parms))
    
        for f in as_completed(futures):
            cpma=cpma.append(f.result())
            
    return(cpma)
    
def cpmaHelp(z,i,parms):
    name=parms['name']
    local=parms['local']
    path=local+name
        
    subprocess.run(['Rscript',path+'R/myCPMA.R',str(i),path])
    
    result=pd.read_csv(path+'cpma/cpma-'+str(i)+'.csv')
    cpma=pd.DataFrame({'Type':'cpma','Value':result['pval']})
    
    return(cpma)
        