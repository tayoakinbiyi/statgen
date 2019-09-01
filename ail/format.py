from os import listdir
from os.path import isfile, join
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed

mypath='/phddata/akinbiyi/scratch/ail'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f[0]=='z']

def fmt(f):
    print(f,flush=True)
    df=np.loadtxt(f,delimiter=',')
    np.savetxt(f,df,delimiter=',',fmt='%1.3f')
    
futures=[]
with ProcessPoolExecutor() as executor: 
    for f in onlyfiles:
        futures.append(executor.submit(fmt,f))
        
np.savetxt('/phddata/akinbiyi/files.csv',onlyfiles,delimiter=',')