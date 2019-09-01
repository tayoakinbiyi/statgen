import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import numpy as np

def genMeans(parms):
    scratchDir=parms['scratchDir']
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitChr=parms['traitChr']
    smallCpu=parms['smallCpu']
        
    #genCorrMean('chr1',parms)
    futures=[]
    with ProcessPoolExecutor(smallCpu) as executor: 
        for trait in traitChr:
            futures.append(executor.submit(genMeansHelp,trait,parms))

        wait(futures,return_when=ALL_COMPLETED)
        
def genMeansHelp(trait,parms): 
    scratchDir=parms['scratchDir']
    snpChr=parms['snpChr']
    
    if os.path.isfile(scratchDir+'mean-'+trait+'.csv'):
        return()
    
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitData=traitData[traitData['chr']==trait]
    snpData=pd.read_csv(scratchDir+'snpData.csv')

    mean=np.zeros([1,traitData.shape[0]])
    std=np.zeros([1,traitData.shape[0]])
    
    K=float(len(snpData))
    
    for snp in snpChr:
        if snp==trait:
            continue
            
        print('for mean/std reading z scores '+snp+' '+trait,flush=True)
        df=np.loadtxt(scratchDir+'z-'+snp+'-'+trait+'.csv',delimiter=',')
        
        if df.shape[1]!=mean.shape[1]:
            print(trait,'wrong shape',df.shape[1],mean.shape[0],flush=True)
            sys.exit(1)
            
        mean+=np.sum(df,axis=0).reshape(1,-1)
        std+=np.sum(df**2,axis=0).reshape(1,-1)
    
    print(trait,flush=True)
    
    mean/=K
    std=((K/(K-1))*(std/K-mean**2))**(.5)
    
    np.savetxt(scratchDir+'mean-'+trait+'.csv',mean,delimiter=',')
    np.savetxt(scratchDir+'std-'+trait+'.csv',std,delimiter=',')
    
    return()
