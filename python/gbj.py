from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil
import subprocess
import os

def gbj(z,parms):       
    cpus=parms['cpus']
    Reps,N=z.shape

    futures=[]
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(int(np.ceil(Reps/np.ceil(Reps/cpus)))):
            futures.append(executor.submit(gbjHelp,z[i*int(np.ceil(Reps/cpus)):min((i+1)*int(np.ceil(Reps/cpus)),Reps)],i,parms))
    
    gbj=pd.DataFrame(dtype='float32')
    fail=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        gbj=gbj.append(result[0])
        fail=fail.append(result[1])
        
    return(gbj,fail)
    
def gbjHelp(z,i,parms):
    delta=parms['delta']
    path=parms['path']+parms['sigName']+'/'
    
    np.savetxt(path+'gbj/z_'+str(i)+'.csv',z,delimiter=',')
    subprocess.run(['Rscript',path+'R/myGBJ.R',str(i),path,str(delta)])
    
    result=pd.read_csv(path+'gbj/gbj_'+str(i)+'.csv')
    gbj=pd.DataFrame({'Type':'gbj','Value':result['Value']})
    fail=pd.DataFrame({'Type':'gbj','Value':result['Fail']})
    
    return(gbj,fail)
        