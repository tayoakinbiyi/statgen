import numpy as np
import pandas as pd
import pdb
from scipy.stats import norm, beta

from ail.statsPython.cpma import *
from ail.opPython.DB import *

def noCorrStats(z,pval,nameParm,parms,folder):
    name=parms['name']
    
    N=pval.shape[1]
    pval=pval[:,0:int(N/2)]
    Reps,d=pval.shape

    Fn=(np.array([range(d)]*Reps)+1)/N
    Fn0=Fn[0]

    hc=[np.sqrt(N)*np.max(((Fn0 -p)/np.sqrt(p*(1-p)))[p>=1/N]) for p in pval]
    bj=np.max(N*D(Fn,pval),axis=1)
    gnull=np.max(-beta.cdf(pval,np.array([range(1,d+1)]*Reps),N-np.array([range(1,d+1)]*Reps)),axis=1)
    score=np.sum(z**2,axis=1)
    
    DBWrite(np.array(hc),name+folder+'hc-'+nameParm,parms,True)
    DBWrite(np.array(bj),name+folder+'bj-'+nameParm,parms,True)
    DBWrite(np.array(gnull),name+folder+'gnull-'+nameParm,parms,True)
    DBWrite(np.array(score),name+folder+'score-'+nameParm,parms,True)

    DBLog('noCorrStats '+nameParm+'\nnoCorrStats score len:min:max \t\t'+str(len(score))+' : '+str(min(score))+' : '+str(max(score))+
          '\nnoCorrStats bj len:min:max \t\t\t'+str(len(bj))+' : '+str(min(bj))+' : '+str(max(bj))+
          '\nnoCorrStats gnull len:min:max \t\t'+str(len(gnull))+' : '+str(min(gnull))+' : '+str(max(gnull))+
          '\nnoCorrStats hc len:min:max \t\t\t'+str(len(hc))+' : '+str(min(hc))+' : '+str(max(hc)),parms)

    print('noCorrStats '+nameParm,flush=True)
    
    return()

def D(u,v):
    return(u*np.log(u/v)+(1-u)*np.log((1-u)/(1-v)))