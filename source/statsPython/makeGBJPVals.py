from opPython.DB import *
from scipy.stats import norm, beta
from statsPython.gbj import *
from ELL.util import memory

import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def makeGBJPVals(parms,z,offDiag):
    memory('makeGBJPVals')
    numCores=parms['numCores']
    
    segLen=int(np.ceil(len(z)/numCores))
    futures=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for core in range(numCores):
            snpRange=np.arange(core*segLen,min((core+1)*segLen,len(z)))
            if len(snpRange)==0:
                continue

            futures+=[executor.submit(gbj,snpRange,z[snpRange],offDiag)]

        Pgbj=[]
        Pghc=[]
        Phc=[]
        Pbj=[]
        PminP=[]
        snpRange=[]
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            _snpRange,_Pgbj,_Pghc,_Phc,_Pbj,_PminP=f.result()
            snpRange+=[_snpRange.reshape(-1,1)]
            Pgbj+=[_Pgbj.reshape(-1,1)]
            Pghc+=[_Pghc.reshape(-1,1)]
            Pbj+=[_PminP.reshape(-1,1)]
            Phc+=[_Phc.reshape(-1,1)]
            PminP+=[_PminP.reshape(-1,1)]                
    
    snpOrd=np.argsort(np.concatenate(snpRange,axis=0).flatten())
    Pgbj=np.concatenate(Pgbj,axis=0)[snpOrd]
    Pghc=np.concatenate(Pghc,axis=0)[snpOrd]
    Phc=np.concatenate(Phc,axis=0)[snpOrd]
    Pbj=np.concatenate(Pbj,axis=0)[snpOrd]
    PminP=np.concatenate(PminP,axis=0)[snpOrd]

    memory('makeGBJPVals')
    
    return(Pgbj,Pghc,Phc,Pbj,PminP)
                          