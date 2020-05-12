import numpy as np
import pdb
import os
from ELL.F import *

def preScoreHelp(kRange,b_score,lamEllByK,b_check):      
    for kInd in range(len(kRange)):
        k=kRange[kInd]
        b_check[0][:,k]=[k,np.mean(b_score[0][:,kInd]<lamEllByK[0,k]),np.mean(b_score[0][:,kInd]>lamEllByK[-1,k])]
        sortOrd=np.argsort(b_score[0][:,kInd])
        b_score[0][sortOrd,kInd]=np.clip(np.searchsorted(lamEllByK[:,k],b_score[0][sortOrd,kInd],side='left'),0,
            lamEllByK.shape[0]-1)   
        b_score[1].flush()
    
    return()

def scoreHelp(b_score,N,nCr,offDiagMeans):    
    Reps,maxD=b_score[0].shape
    for row in np.arange(Reps):
        for trait in np.arange(b_score[0].shape[1]):
            b_score[0][row,trait]=F(N,b_score[0][row,trait],trait,trait,nCr,offDiagMeans)
    
    return()

