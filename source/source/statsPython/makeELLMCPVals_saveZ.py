from opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb
import random
from scipy.stats import norm
from statsPython.f_ELL import *

def makeELLMCPVals(parms):
    transOnly=parms['transOnly']
    ellDSet=parms['ellDSet']
    traitChr=parms['traitChr']
    muEpsRange=[[0,0]]+parms['muEpsRange']
    
    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
    N=LZCorr.shape[0]

    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)
    traitData=traitData[traitData['chr'].isin(traitChr)]
    traitLoc=traitData['chr'].values.flatten()
    
    if not transOnly:
        refs=genRef(parms,LZCorr)
        
    for ind in range(len(muEpsRange)):                
        muEps=muEpsRange[ind]
        mu=muEps[0]
        eps=muEps[1]
        snp=len(parms['SnpSize'])+ind

        if transOnly:
            refs=genRef(parms,LZCorr[traitLoc!=snp,traitLoc!=snp])
        
        for dParm in ellDSet:     
            ref=np.sort(refs[:,0:int(dParm*N)].min(axis=1))
            stat=np.loadtxt('stats/'+str(dParm)+'-'+str(snp),delimiter='\t')
            statOrd=np.argsort(stat)
            pval=(np.searchsorted(ref,stat[statOrd])/(len(ref)+1))[np.argsort(statOrd)]
            
            DBLog(str(dParm)+'-'+str(snp)+' stat ['+str(len(stat))+','+str(np.min(stat))+
                  ','+str(np.mean(stat))+','+str(np.max(stat))+'] pval ['+str(len(pval))+
                  ','+str(np.min(pval))+','+str(np.mean(pval))+','+str(np.max(pval))+'] ref ['+str(len(ref))+
                  ','+str(np.min(ref))+','+str(np.mean(ref))+','+str(np.max(ref))+']')
            
            np.savetxt('pvals/ell_'+str(dParm)+'_MC-'+str(snp),pval,delimiter='\t')

            print('finished '+str(dParm)+'- snp: '+str(snp),flush=True)

    return(refs)

def genRef(parms,L):
    ellDSet=parms['ellDSet']
    RefReps=parms['RefReps']

    N=L.shape[0]
    
    z=np.matmul(norm.rvs(size=[RefReps,N]),L.T)  
    np.savetxt('score/mc',z,delimiter='\t')
    pval=np.sort(2*norm.sf(np.abs(z))) 
    
    d=int(N*max(ellDSet))
    
    stats=f_ELL(pval,d,N,parms)

    return(stats)
