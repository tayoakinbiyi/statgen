import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

def genCorrMats(parms):
    scratchDir=parms['scratchDir']
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitChr=parms['traitChr']
    cpus=10
    
    print('gen corrMats')
    
    for i in range(len(traitChr)):
        for j in range(i,len(traitChr)):
            genCorrMatsHelp(traitChr[i],traitChr[j],parms)


def genCorrMatsHelp(trait_i,trait_j,parms): 
    scratchDir=parms['scratchDir']
    snpChr=parms['snpChr']

    traitData=pd.read_csv(scratchDir+'traitData.csv')
    snpData=pd.read_csv(scratchDir+'snpData.csv')
    K=float(len(snpData))
    
    if os.path.isfile(scratchDir+'corr-'+trait_i+'-'+trait_j+'.csv'):
        return
    else:
        np.savetxt(scratchDir+'corr-'+trait_i+'-'+trait_j+'.csv',np.array([]),delimiter=',')
    
    if trait_i==trait_j:     
        mean=np.loadtxt(scratchDir+'mean-'+trait_i+'.csv',delimiter=',').T
        std=np.loadtxt(scratchDir+'std-'+trait_i+'.csv',delimiter=',').T
        
        traitData=traitData[traitData['chr']==trait_i]    
        corr=np.zeros([traitData.shape[0],traitData.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(parms['cpu']) as executor: 
            for snp in snpChr:
                if snp==trait_i:
                    continue
                
                futures.append(executor.submit(sameTrait,scratchDir,snp,trait_i,mean,std))

            for f in as_completed(futures):
                corr+=f.result()        
        
    else:
        mean_i=np.loadtxt(scratchDir+'mean-'+trait_i+'.csv',delimiter=',').T
        std_i=np.loadtxt(scratchDir+'std-'+trait_i+'.csv',delimiter=',').T

        mean_j=np.loadtxt(scratchDir+'mean-'+trait_j+'.csv',delimiter=',').T
        std_j=np.loadtxt(scratchDir+'std-'+trait_j+'.csv',delimiter=',').T

        traitData_i=traitData[traitData['chr']==trait_i]    
        traitData_j=traitData[traitData['chr']==trait_j]    

        corr=np.zeros([traitData_i.shape[0],traitData_j.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(parms['cpu']) as executor: 
            for snp in snpChr:
                if snp==trait_i or snp==trait_j:
                    continue

                futures.append(executor.submit(diffTrait,scratchDir,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j))
    
            for f in as_completed(futures):
                corr+=f.result()        

    print('writing corr mat '+trait_i+' - '+trait_j,flush=True)
    
    np.savetxt(scratchDir+'corr-'+trait_i+'-'+trait_j+'.csv',(corr/K).astype('float16'),delimiter=',')

def diffTrait(scratchDir,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j):
    df_i=np.loadtxt(scratchDir+'z-'+snp+'-'+trait_i+'.csv',delimiter=',')
    df_j=np.loadtxt(scratchDir+'z-'+snp+'-'+trait_j+'.csv',delimiter=',')

    df_i=(df_i-mean_i)/std_i
    df_j=(df_j-mean_j)/std_j

    return(np.matmul(df_i.T,df_j))
    
def sameTrait(scratchDir,snp,trait,mean,std):
    df=np.loadtxt(scratchDir+'z-'+snp+'-'+trait+'.csv',delimiter=',')
    df=(df-mean)/std

    return(np.matmul(df.T,df))
    