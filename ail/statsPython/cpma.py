import pandas as pd
import numpy as np
import pdb
import subprocess
from ail.opPython.DB import *
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

def cpma(pval,nameParm,folder,parms):       
    name=parms['name']
    
    N=pval.shape[1]
    
    cpmaString='''
        cpma <- function(pvals,log=T,zero.val=NA) {

            ## this function tests deviation from the expected exponential
            ## behaviour of -log(p) for a set of associations to a SNP.
            ## modelled on suggestions from Chris Wallace/David Clayton, it's an
            ## implementation of a method proposed by Ben Voight

            ## Chris Cotsapas 2011, based on previous code written 2009

            ## this version avoids multiplication which eventually converges to zero for
            ## large p value series. Instead it transforms to log space and adds.

            ## internal function to compute likelihood of exponential distribution at a rate lambda
            int.exp.fn <- function(x,lambda=1,epsilon=0.001) {
            return( exp(-lambda * (x-epsilon)) - exp(-lambda * (x+epsilon)) )
            }

            pvals[pvals==0] <- zero.val

            borked.ix <- ( is.na(pvals) | is.infinite(pvals) )

            pvals <- pvals[!borked.ix]

            if(log) {
            pvals <- -log(pvals)
            }

            ## return the log likelihood ratio of -log(p) being exponentially
            ## distributed (cf biased exponential). Effectively, test if
            ## exponential decay parameter == 1

            ## this should be chi-sq,df=1
            #  abs(-2 * log(prod(int.exp.fn(pvals,1/mean(pvals))) / prod(int.exp.fn(pvals,1)) ))
            p.obs <-  sum(log(int.exp.fn(pvals,1/mean(pvals))))
            p.exp <- sum(log(int.exp.fn(pvals,1)))

            stat <- -2 * (p.obs-p.exp)

            return(abs(stat))
        }
    '''

    cpma=SignatureTranslatedAnonymousPackage(cpmaString,'cpma')    

    stats=[]
    for row in pval:
        vec=ro.FloatVector(tuple(row))
        stats+=[cpma.cpma(vec)]
    
    stats=np.array(stats).flatten()
    
    DBLog('cpma '+nameParm+'\tlen:min:max \t'+str(len(stats))+' : '+str(min(stats))+' : '+str(max(stats)),parms)
    
    DBWrite(stats,name+folder+'cpma-'+nameParm,parms,True)
    
    print('cpma '+nameParm,flush=True)
    
    return()
        