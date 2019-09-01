import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import numpy as np
from ail.genPython.makePSD import *

def genCorr(trait,parms):
    scratchDir=parms['scratchDir']
    traitChr=parms['traitChr']
    
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitData=traitData[traitData['chr']!=trait]    

    if os.path.isfile(scratchDir+'corr-'+trait+'.csv'):
        corr=np.loadtxt(scratchDir+'corr-'+trait+'.csv',delimiter=',')
        return(corr)
    
    corr=np.empty([traitData.shape[0],traitData.shape[0]])

    if os.path.isfile(scratchDir+'corr-'+trait+'.csv'):
        return()

    for i in range(len(traitChr)):
        for j in range(i,len(traitChr)):
            if traitChr[i]==trait or traitChr[j]==trait:
                continue

            print('loading '+traitChr[i]+'-'+traitChr[j],flush=True)               

            xLoc=np.arange(len(traitData))[(traitData['chr']==traitChr[i]).values.flatten()]
            yLoc=np.arange(len(traitData))[(traitData['chr']==traitChr[j]).values.flatten()]

            df=np.loadtxt(scratchDir+'corr-'+traitChr[i]+'-'+traitChr[j]+'.csv',delimiter=',')
            #pdb.set_trace()
            corr[xLoc.reshape(-1,1),yLoc]=df

            if i!=j:
                corr[yLoc.reshape(-1,1),xLoc]=df.T
    
        np.fill_diagonal(corr, 1)
        
    L=makePSD(corr)
    corr=np.matmul(L,L.T)  
    
    np.savetxt(scratchDir+'corr-'+trait+'.csv',corr,delimiter=',')
    print('return')
    return(corr)
