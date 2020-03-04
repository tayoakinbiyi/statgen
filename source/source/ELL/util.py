import numpy as np
import psutil
import mmap
import os
import pdb
import subprocess

def nCr(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def memory(label):
    mem=np.round(np.array([[proc.memory_info().rss/2**20,proc.memory_info().shared/2**20,proc.memory_percent()] for 
        proc in [psutil.Process()]]),2)

    print(label+' rss(mb) {}, shared {}, memPct {}'.format(mem[0,0],mem[0,1],mem[0,2]),flush=True)

    return()

def bufCreate(name,shape):
    numPages=int(np.ceil(np.prod(shape)*8/mmap.PAGESIZE))
    arrSize=int(numPages*mmap.PAGESIZE)
    path='/dev/shm/'+name
    
    if os.path.exists(path):
        os.remove(path)
        
    fd=os.open(path, os.O_CREAT | os.O_RDWR)
    assert os.write(fd, b'\x00' * arrSize) == arrSize

    buf = mmap.mmap(fd, arrSize, mmap.MAP_SHARED)
        
    arr=np.ndarray(shape, dtype='float64', buffer=buf)
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