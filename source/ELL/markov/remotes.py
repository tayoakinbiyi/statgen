from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import numpy as np
from scipy.stats import norm
import pdb
import os

def markovHelp(repRange,b_markov,stats,lamEllByK,ellGrid,d,N,offDiagVec):    
    gbj=importr('GBJ')
    
    row=np.zeros(N)
    for rep in repRange:
        row[0:d]=-norm.ppf(lamEllByK[np.searchsorted(ellGrid,stats[rep]),0:d]/2)
        row[d:]=row[d-1]

        bounds=ro.FloatVector(row[::-1])
        b_markov[0][rep]=gbj.ebb_crossprob_cor_R(d=N, bounds=bounds, correlations=offDiagVec)[0]
        b_markov[1].flush()
    
    return()