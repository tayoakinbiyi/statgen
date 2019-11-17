import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import numpy as np

from ail.opPython.DB import *

def genMeans(parms):
    name=parms['name']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    smallNumCores=parms['smallNumCores']
    
    #genMeansHelp('chr1',parms,nameParm)
    traitDF=[]
    for trait in traitChr:
        snpDF=[]
        for snp in snpChr:
            snpDF+=[DBRead(name+'score/'+trait,parms)]
        traitDF+=[np.concatenate(snpDF,axis=0)]
    traitDF=np.concatenate(traitDF,axis=1)

    mean=np.mean(df,axis=0).reshape(1,-1)
    mean2=np.mean(df**2,axis=0).reshape(1,-1)
    mean4=np.mean(df**4,axis=0).reshape(1,-1)
    std=np.std(df,axis=0).reshape(1,-1)

    DBWrite(mean,name+'corr/mean',parms)
    DBWrite(mean2,name+'corr/mean2',parms)
    DBWrite(mean4,name+'corr/mean4',parms)
    DBWrite(std,name+'corr/std',parms)
        
    return()

