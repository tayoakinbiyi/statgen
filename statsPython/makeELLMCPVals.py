from ail.opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb
import random
from scipy.stats import norm
from ail.statsPython.f_ELL import *

def makeELLMCPVals(parms):
    transOnly=parms['transOnly']
    muEpsRange=parms['muEpsRange']
    ellDSet=parms['ellDSet']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    
    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')

    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)
    traitData=traitData[traitData['chr'].isin(traitChr)]
    traitLoc=traitData['chr'].values.flatten()
      
    if not transOnly:
        ref=genRef(parms,LZCorr)
        
    for snp in snpChr:
        if transOnly:
            ref=genRef(parms,LZCorr[traitLoc!=snp,traitLoc!=snp])
        
        for dParm in ellDSet:                        
            stat=np.loadtxt('stats/'+str(dParm)+'-'+str(snp),delimiter='\t')
            statOrd=np.argsort(stat)
            pval=(1-np.searchsorted(ref[dParm],stat[statOrd])/(len(ref[dParm])+1))[np.argsort(statOrd)]
            
            DBLog(str(dParm)+'-'+str(snp)+' stat ['+str(len(stat))+','+str(np.min(stat))+
                  ','+str(np.mean(stat))+','+str(np.max(stat))+'] pval ['+str(len(pval))+
                  ','+str(np.min(pval))+','+str(np.mean(pval))+','+str(np.max(pval))+']',parms)
            
            np.savetxt('pvals/ell_'+str(dParm)+'_MC-'+str(snp),pval,delimiter='\t')

            print('finished '+str(dParm)+'- snp: '+str(snp),flush=True)
        
    return()

def genRef(parms,L):
    ellDSet=parms['ellDSet']
    RefReps=parms['RefReps']

    N=L.shape[0]
    
    z=-np.sort(-np.abs(np.matmul(norm.rvs(size=[RefReps,N]),L.T)))   
    pval=2*norm.sf(np.abs(z))        
    
    d=int(N*max(ellDSet))
    
    stats=f_ELL(pval,d,N,parms)

    df={}
    for dParm in ellDSet:
        df[dParm]=np.amin(stats[:,0:int(dParm*N)],axis=1)
    
    return(df)
