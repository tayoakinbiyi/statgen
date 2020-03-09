import pandas as pd
import numpy as np
import pdb
import subprocess
from opPython.DB import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

def gbj(snpRange,z,offDiag): 
    gbj=importr('GBJ')
    
    Reps,N=z.shape
        
    Pgbj=np.arange(Reps).astype(float)
    Pghc=np.arange(Reps).astype(float)
    Phc=np.arange(Reps).astype(float)
    Pbj=np.arange(Reps).astype(float)
    PminP=np.arange(Reps).astype(float)
    
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    for row in range(Reps):
        z_vec=ro.FloatVector(tuple(z[row]))
        
        Pbj[row]=gbj.BJ(test_stats=z_vec,  pairwise_cors=offDiagVec)[1][0]
        Pgbj[row]=gbj.GBJ(test_stats=z_vec, pairwise_cors=offDiagVec)[1][0]
        Pghc[row]=gbj.GHC(test_stats=z_vec, pairwise_cors=offDiagVec)[1][0]
        Phc[row]=gbj.HC(test_stats=z_vec, pairwise_cors=offDiagVec)[1][0]
        PminP[row]=gbj.minP(test_stats=z_vec, pairwise_cors=offDiagVec)[1][0]
    
    return(snpRange,Pbj,Pghc,Phc,Pbj,PminP)
        