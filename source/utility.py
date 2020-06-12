import os
import json
import pdb
import subprocess
from zipfile import ZipFile
import pprint
import shutil
import sys
import numpy as np
import socket
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from scipy.stats import beta
import psutil
import mmap
import csv
import io

from plotPython.myQQ import *

def setupFolders():
    name=sys.argv[0][:-3]
    
    if not os.path.exists(name):
        os.mkdir(name)
        os.chdir(name)
    else:
        os.chdir(name)    
           
    return()

def createDiagnostics(seed=None):
    subprocess.call(['rm','-rf','diagnostics/'])
    subprocess.call(['mkdir','diagnostics'])
    shutil.make_archive('diagnostics/python', "zip", '../source')
    subprocess.call(['cp','../'+sys.argv[0],'diagnostics/'+sys.argv[0]])
    
    if seed is None:
        seed=int(np.random.randint(1e6))
        
    np.random.seed(seed)
    log('random seed {}'.format(seed))

    return()
       
def git(msg): 
    hostname=socket.gethostname()
        
    local=os.getcwd()+'/'
    os.chdir('../'+hostname)
    subprocess.call(['git','fetch','--all'])
    subprocess.call(['git','reset','--hard','HEAD'])
    for file in [file for file in os.listdir() if file!='.git']:
        subprocess.call(['git','rm','-r',file])
    for file in os.listdir(local+'diagnostics'):
        if os.stat(local+'diagnostics/'+file).st_size<50*1024**2:
            subprocess.call(['cp','-rf',local+'diagnostics/'+file,file])  
    subprocess.call(['git','add','-A'])
    subprocess.call(['git','commit','-m',msg])
    subprocess.call(['git','push','-f','origin','archive'])
    os.chdir(local)
    
    return()
    
def log(msg): 
    msg=pprint.pformat(msg,compact=True)
    
    print(msg,flush=True)

    with open('diagnostics/log','a+') as f:
        f.write(msg+'\n')
                
    return()

def makeL(df,inv=False):
    U,D,Vt=np.linalg.svd(df)
    if inv:
        D=1/D
    L=np.matmul(U,np.diag(np.sqrt(D)))

    return(L)

def memory(label):
    mem=np.round(np.array([[proc.memory_info().rss/2**20,proc.memory_info().shared/2**20,proc.memory_percent()] for 
        proc in [psutil.Process()]]),2)

    print(label+' rss(mb) {}, shared {}, memPct {}'.format(mem[0,0],mem[0,1],mem[0,2]),flush=True)

    return()

def bufCreate(name,shape,dtype='float64'):
    numPages=int(np.ceil(np.prod(shape)*np.dtype(dtype).itemsize/mmap.PAGESIZE))
    arrSize=int(numPages*mmap.PAGESIZE)
    path='/dev/shm/'+name
    
    if os.path.exists(path):
        os.remove(path)
        
    fd=os.open(path, os.O_CREAT | os.O_RDWR)
    assert os.write(fd, b'\x00' * arrSize) == arrSize

    buf = mmap.mmap(fd, arrSize, mmap.MAP_SHARED)
        
    arr=np.ndarray(shape, dtype=dtype, buffer=buf)
    arr[:]=0

    return((arr,buf,fd,name))

def bufClose(buf):
    arr=buf[0].copy()
    buf[1].close()
    os.close(buf[2])
    subprocess.call(['rm','/dev/shm/'+buf[3]])
    
    return(arr)

def remote(*args):
    pid=os.fork()
    
    if pid==0: # child
        tuple(args)[0](*args[1:])
        exit()
    else:
        return(pid)
    
def arrString(arr):
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerows(arr.reshape(len(arr),-1))
    
    return(si.getvalue())
