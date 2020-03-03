import os
import numpy as np
import pandas as pd
import pdb

from genPython.makePSD import *

def genLZCorr(parms):
    transOnly=parms['transOnly']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']   
    
    traitData=pd.read_csv('ped/traitData',sep='\t',header=0,index_col=None)
            
    snpData=pd.read_csv('ped/snpData',sep='\t',header=0,index_col=None)
    
    snpLoc=snpData['chr'][snpData['chr'].isin(snpChr)].values.flatten()
    traitLoc=traitData['chr'][traitData['chr'].isin(traitChr)].values.flatten()

    traitDF=np.full([len(snpLoc),len(traitLoc)],np.nan)

    for trait in traitChr:
        for snp in snpChr:
            if trait==snp and transOnly:
                continue
            
            xLoc=np.arange(len(snpLoc))[snpLoc==snp].reshape(-1,1)
            yLoc=np.arange(len(traitLoc))[traitLoc==trait].reshape(1,-1)
            traitDF[xLoc,yLoc]=np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t')

    corr=np.ma.corrcoef(traitDF,rowvar=False,allow_masked=True).data
    LZCorr=makePSD(corr)
    N=LZCorr.shape[1]

    print('writing corr',flush=True)
    np.savetxt('LZCorr/LZCorr',LZCorr,delimiter='\t')
    offDiag=corr[np.triu_indices(N,1)]
        
    return(LZCorr,offDiag)