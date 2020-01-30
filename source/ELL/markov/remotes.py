import ray
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import numpy as np
from scipy.stats import norm
import pdb

@ray.remote
def markovHelp(repRange,markov,stats,lamEllByK,ellGrid,dList,N,offDiagVec):    
    gbj=importr('GBJ')
    
    row=np.zeros(N)
    for dInd in range(len(dList)):
        d=dList[dInd]
        
        for rep in repRange:
            row[0:d]=-norm.ppf(lamEllByK[np.searchsorted(ellGrid,stats[rep,dInd]),0:d]/2)
            row[d:]=row[d-1]

            bounds=ro.FloatVector(row[::-1])
            markov[rep,dInd]=gbj.ebb_crossprob_cor_R(d=N, bounds=bounds, correlations=offDiagVec)[0]
    
    return()