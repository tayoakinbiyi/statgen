from opPython.DB import *
from statsPython.f_ELL import *

import numpy as np
from scipy.stats import norm

def genELL(parms):
    transOnly=parms['transOnly']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    muEpsRange=[[0,0]]+parms['muEpsRange']
                    
    print('genELL',flush=True)
        
    ellDSet=parms['ellDSet']
    
    for ind in range(len(muEpsRange)):                
        muEps=muEpsRange[ind]
        mu=muEps[0]
        eps=muEps[1]
        snp=len(parms['SnpSize'])+ind

        if transOnly:
            traitChr=[trait for trait in parms['traitChr'] if trait!=snp]
        else:
            traitChr=parms['traitChr']

        df=[]
        for trait in traitChr:
            df+=[np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t')]

        z=-np.sort(-np.abs(np.concatenate(df,axis=1)))
        pval=2*norm.sf(np.abs(z))
        Reps,N=pval.shape
        d=int(N*max(ellDSet))

        stats=f_ELL(pval,d,N,parms)
        
        for dParm in ellDSet:
            d=int(dParm*N)
            np.savetxt('stats/'+str(dParm)+'-'+str(snp),stats[:,0:d].min(axis=1),delimiter='\t')

    return(stats)
    