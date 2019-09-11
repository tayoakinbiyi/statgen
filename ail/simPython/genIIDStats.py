from ail.opPython.DB import *
from ail.statsPython.fitMCMCStats import *
import numpy as np

def genIIDStats(parms):    
    IIDReps=parms['IIDReps']
    ellDelta=parms['ellDelta']
    
    LZCorr=DBRead(name+'process/LZCorr-all',parms,toPickle=True)
    N=DBRead(name+'process/traitData',parms,toPickle=True).shape[0]
    d=int(N*ellDelta)

    z=np.matmul(norm.rvs(size=[IIDReps,N]),LZCorr.T)
    z=-np.sort(-np.abs(z))[:,0:d]
        
    stats=fitMCMCStats(z,N,parms)
    DBWrite(stats,name+'sim/IIDZScores',parms,toPickle=True)
    
    return()