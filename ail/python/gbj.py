from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil
import subprocess

def gbj(z,sigName):       
    Reps,N=z.shape

    M=multiprocessing.cpu_count()
    
    i=0
    gbjHelp(z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)],i,sigName)
    futures=[]
    with ProcessPoolExecutor() as executor: 
        for i in range(int(np.ceil(Reps/np.ceil(Reps/M)))):
            futures.append(executor.submit(gbjHelp,z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)],i,sigName))
    
    gbj=pd.DataFrame(dtype='float32')
    fail=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        gbj=gbj.append(result[0])
        fail=fail.append(result[1])
        
    return(gbj,fail)
    
def gbjHelp(z,i,sigName):  
    name='/project/abney/ail/gbj/'
    np.savetxt(name+'z_'+str(i)+'.csv',z,delimiter=',')
    subprocess.run(['Rscript','/home/akinbiyi/ail/ail/R/myGBJ.R',str(i),sigName])
    
    result=pd.read_csv(name+'gbj_'+str(i)+'.csv')
    gbj=pd.DataFrame({'Type':'gbj','Value':result['Value']})
    fail=pd.DataFrame({'Type':'gbj','Value':result['Fail']})
    
    return(gbj,fail)
        