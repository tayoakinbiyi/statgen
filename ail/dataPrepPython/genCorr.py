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

def genCorr(trait,parms):
    name=parms['name']
    traitChr=parms['traitChr']
    
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)
    traitData=traitData[traitData['chr']!=trait]    

    if DBIsFile(name+'corr/','LZCorr-'+trait,parms):
        return()
    
    corr=np.empty([traitData.shape[0],traitData.shape[0]])

    for i in range(len(traitChr)):
        for j in range(i,len(traitChr)):
            if traitChr[i]==trait or traitChr[j]==trait:
                continue

            print('loading cor mats '+traitChr[i]+' - '+traitChr[j],flush=True)               

            xLoc=np.arange(len(traitData))[(traitData['chr']==traitChr[i]).values.flatten()]
            yLoc=np.arange(len(traitData))[(traitData['chr']==traitChr[j]).values.flatten()]

            df=DBRead(name+'corr/corr-'+traitChr[i]+'-'+traitChr[j],parms,toPickle=True)
            corr[xLoc.reshape(-1,1),yLoc]=df

            if i!=j:
                corr[yLoc.reshape(-1,1),xLoc]=df.T
    
        np.fill_diagonal(corr, 1)
        
    LZCorr=makePSD(corr)
    
    print('writing corr',flush=True)
    DBWrite(LZCorr,name+'corr/LZCorr-'+trait,parms,toPickle=True)

    return()
