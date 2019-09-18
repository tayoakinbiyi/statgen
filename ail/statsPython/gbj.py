from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil
import subprocess
import os

def gbj(numSegs,gbjName,parms):       
    cpus=parms['cpus']
    name=parms['name']
    Reps,d=z.shape

    futures=[]
    gbj=pd.DataFrame(dtype='float32')
    fail=pd.DataFrame(dtype='float32')
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(numSegs):
            futures.append(executor.submit(gbjHelp,i,parms))
    
        for f in as_completed(futures):
            result=f.result()
            gbj=gbj.append(result[0])
            fail=fail.append(result[1])
        
    DBWrite(fail,name+'gbj/'+gbjName,parms,toPickle=False)
    
    return(gbj)
    
def gbjHelp(i,parms):
    name=parms['name']
    local=parms['local']
    path=local+name
        
    subprocess.run(['Rscript',path+'R/myGBJ.R',str(i),path])
    
    result=pd.read_csv(path+'gbj/gbj-'+str(i)+'.csv')
    gbj=pd.DataFrame({'Type':'gbj','Value':result['pval']})
    fail=pd.DataFrame({'Type':'gbj','Value':result['Fail']})
    
    return(gbj,fail)
        