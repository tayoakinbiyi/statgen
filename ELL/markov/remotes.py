import ray
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

@ray.remote
def markovHelp(markov,repRange,stats,zEllByK,dList,N,offDiagVec):    
    gbj=importr('GBJ')
        
    row=np.arange(N)
    for dInd in range(len(dList)):
        d=dList[dInd]
        
        for repInd in range(len(repRange)):
            row[0:d]=zEllByK[stats[repRange[repInd]],0:d]
            row[d:]=row[d-1]

            bounds=ro.FloatVector(row[::-1])
            markov[repInd,dInd]=gbj.ebb_crossprob_cor_R(d=N, bounds=bounds, correlations=offDiagVec)[0]
    
    return()