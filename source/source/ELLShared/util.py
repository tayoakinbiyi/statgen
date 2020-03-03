import numpy as np
import psutil

def nCr(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def memory(label):
    memAmt=np.round(np.sum(np.array([(proc.memory_info().rss-proc.memory_info().shared)/1024**2 for proc in psutil.Process(
        ).children(recursive=True)]+[(psutil.Process().memory_info().rss-psutil.Process().memory_info().shared)/1024**2])),2)
    memPct=np.round(np.sum(np.array([proc.memory_percent() for proc in 
                psutil.Process().children(recursive=True)]+[psutil.Process().memory_percent()])),2)

    print(label+' memAmt(mb), memPct '+str(memAmt)+' , '+str(memPct),flush=True)

    return()

def memCreate(name,shape):
    arrSize=int(np.array(1,dtype='float64').itemsize*np.prod(shape))
    numPages=int(np.ceil(arrSize/mmap.PAGESIZE)*arrSize)
    
    fd = os.open('/tmp/shm/'+name, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
    assert os.write(fd, '\x00' * arrSize) == arrSize

    buf = mmap.mmap(fd, numPages, mmap.MAP_SHARED, mmap.PROT_WRITE)
    arr=np.ndarray([len(ellGrid),maxD], dtype='float64', buffer=buf)
        
    arr[:]=0.0

    return(buf)

