import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import numpy as np

from ail.genPython.makePSD import *
from ail.opPython.DB import *

def genLZCorr(parms):
    name=parms['name']
    traitChr=parms['traitChr']
    
    traitData=DBRead(name+'process/traitData',parms)

    corr=np.empty([traitData.shape[0],traitData.shape[0]])

    df=[]
    for trait in traitChr:
        snpDF=[]
        for snp in snpChr:
            snpDF+=[DBRead('score/'+snp+'-'+trait,parms)]
        df+=[np.concatenate(snpDF,axis=1)]
        
    df=np.concatenate(df,axis=0)

    corr=np.corrcoef(df,rowvar=False)
    
    np.fill_diagonal(corr, 1)
        
    LZCorr=makePSD(corr)
    
    print('writing corr',flush=True)
    DBWrite(LZCorr,name+'LZCorr/LZCorr',parms)
    
    offDiag=corr[np.triu_indices(len(corr),1)].flatten()

    DBWrite(offDiag,name+'offDiag/offDiag',parms,True)

    return()
