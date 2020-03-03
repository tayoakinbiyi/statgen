import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import numpy as np

from ail.opPython.DB import *

def genSnpMeans(parms,nameParm=''):
    name=parms['name']
    snpChr=parms['snpChr']
    smallNumCores=parms['smallNumCores']
    
    if nameParm !='':
        nameParm='-'+nameParm
        
    #genMeansHelp('chr1',parms)
    futures=[]
    with ProcessPoolExecutor(smallNumCores) as executor: 
        for snp in snpChr:
            if DBIsFile(name+'holds','snpMean-'+snp+nameParm,parms):
                continue

            DBWrite(np.array([]),name+'holds/snpMean-'+snp+nameParm,parms,True)
            futures.append(executor.submit(genSnpMeansHelp,snp,parms,nameParm))

        wait(futures,return_when=ALL_COMPLETED)
        
def genSnpMeansHelp(snp,parms,nameParm): 
    name=parms['name']
    snpChr=parms['snpChr']
    cisMean=parms['cisMean']
    
    snpData=DBRead(name+'process/snpData',parms,toPickle=True)
    snpData=snpData.loc[snpData['chr'].values.flatten()==snp]
    snpData=DBRead(name+'process/snpData',parms,toPickle=True)

    mean=np.zeros([snpData.shape[0],1])
    mean2=np.zeros([snpData.shape[0],1])
    mean4=np.zeros([snpData.shape[0],1])
    
    K=0
    
    for trait in traitChr:
        if (snp==trait) and not cisMean:
            continue
            
        print('for mean/std reading z scores '+snp+' '+trait+nameParm,flush=True)
        df=DBRead(name+'score/p-'+snp+'-'+trait+nameParm,parms,True)
        
        mean+=np.sum(df,axis=1).reshape(-1,1)
        mean2+=np.sum(df**2,axis=1).reshape(-1,1)
        mean4+=np.sum(df**4,axis=1).reshape(-1,1)
        K+=df.shape[1]
    
    print('writing genMeans '+trait,flush=True)
    
    mean/=K
    mean2/=K
    mean4/=K
    
    DBWrite(mean,name+'corr/snpMean-'+snp+nameParm,parms,toPickle=True)
    DBWrite(mean2,name+'corr/snpMean2-'+snp+nameParm,parms,toPickle=True)
    DBWrite(mean4,name+'corr/snpMean4-'+snp+nameParm,parms,toPickle=True)
    
    return()
