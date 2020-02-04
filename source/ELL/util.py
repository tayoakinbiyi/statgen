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
