import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb
from ELL.util import *
from multiprocessing import cpu_count
import rpy2.robjects as ro
import time
from utility import *
import subprocess
import os
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from scipy.stats import norm

def cpma(numCores,z): 
    numSnps,numTraits=z.shape
    
    t0=time.time()
        
    b_time=bufCreate('time',[numCores])
    b_cpma=bufCreate('res',[numSnps])

    t1=time.time()
    
    pids=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(cpmaHelp,core,snpRange,z,b_time,b_cpma)]
        
    for pid in pids:
        os.waitpid(0, 0)
        
    return(bufClose(b_cpma),(np.sum(bufClose(b_time))+(t1-t0))/60)

def cpmaHelp(core,snpRange,z,b_time,b_cpma):  
    t0=time.time()
    
    N=z.shape[1]
    pi=2*norm.sf(np.abs(z[snpRange]))
    
    cpmaString='''
        cpma <- function(pvals,log=T,zero.val=NA) {
          ##  this function tests deviation from the expected exponential
          ## behaviour of -log(p) for a set of associations to a SNP.
          ## modelled on suggestions from Chris Wallace/David Clayton, it's an
          ## implementation of a method proposed by Ben Voight

          ## Chris Cotsapas 2011, based on previous code written 2009

          ## this version avoids multiplication which eventually converges to zero for
          ## large p value series. Instead it transforms to log space and adds.

          ## added a bit to deal with 'signing' the statistic for directionality to 'non-random'ness   

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

          ## sign the stat depending on directionality   
          if (1/mean(pvals) < 1) {
             stat <- -1 * stat;
          } 

          return(stat)
        }
    '''

    cpma=SignatureTranslatedAnonymousPackage(cpmaString,'cpma')    

    stats=[]
    for row in pi:
        vec=ro.FloatVector(tuple(row))
        stats+=[cpma.cpma(vec)]
    
    b_cpma[0][snpRange]=-np.array(stats).flatten()
    b_cpma[1].flush()
    
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    b_time[1].flush()
        
    return()