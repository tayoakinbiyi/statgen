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
    smallNumCores=parms['smallNumCores']
    
    #genMeansHelp('chr1',parms,nameParm)
    for trait in traitChr:
        df=DBRead(name+'score/p-'+trait,parms,True)

        mean=np.mean(df,axis=0).reshape(1,-1)
        mean2=np.mean(df**2,axis=0).reshape(1,-1)
        mean4=np.mean(df**4,axis=0).reshape(1,-1)
        std=np.std(df,axis=0).reshape(1,-1)

        DBWrite(mean,name+'corr/mean-'+trait,parms,True)
        DBWrite(mean2,name+'corr/mean2-'+trait,parms,True)
        DBWrite(mean4,name+'corr/mean4-'+trait,parms,True)
        DBWrite(std,name+'corr/std-'+trait,parms,True)
        
    return()

