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
    smallCpu=parms['smallCpu']
        
    genMeansHelp('chr1',parms)
    futures=[]
    with ProcessPoolExecutor(smallCpu) as executor: 
        for trait in traitChr:
            if DBIsFile(name+'corr','mean-'+trait,parms):
                continue

            futures.append(executor.submit(genMeansHelp,trait,parms))

        wait(futures,return_when=ALL_COMPLETED)
        
def genMeansHelp(trait,parms): 
    pdb.set_trace()
    name=parms['name']
    snpChr=parms['snpChr']
    cisMean=parms['cisMean']
    
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)
    traitData=traitData.loc[traitData['chr'].values.flatten()==trait]
    snpData=DBRead(name+'process/snpData',parms,toPickle=True)

    mean=np.zeros([1,traitData.shape[0]])
    mean2=np.zeros([1,traitData.shape[0]])
    mean4=np.zeros([1,traitData.shape[0]])
    
    K=0
    
    for snp in snpChr:
        if (snp==trait) and not cisMean:
            continue
            
        print('for mean/std reading z scores '+snp+' '+trait,flush=True)
        df=DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True)
        
        mean+=np.sum(df,axis=0).reshape(1,-1)
        mean2+=np.sum(df**2,axis=0).reshape(1,-1)
        mean4+=np.sum(df**4,axis=0).reshape(1,-1)
        K+=df.shape[0]
    
    print('writing genMeans '+trait,flush=True)
    
    mean/=K
    mean2/=K
    mean4/=K
    std=((K/(K-1))*(mean2-mean**2))**(.5)
    
    DBWrite(mean,name+'corr/mean-'+trait,parms,toPickle=True)
    DBWrite(mean2,name+'corr/mean2-'+trait,parms,toPickle=True)
    DBWrite(mean4,name+'corr/mean4-'+trait,parms,toPickle=True)
    DBWrite(std,name+'corr/std-'+trait,parms,toPickle=True)
    
    return()
