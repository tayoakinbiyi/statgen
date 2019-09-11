import pandas as pd
import pdb
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from ail.opPython.DB import *

def genCorrMats(parms):
    name=parms['name']
    traitChr=parms['traitChr']
    
    print('gen corrMats')
    
    for i in range(len(traitChr)):
        for j in range(i,len(traitChr)):
            if DBIsFile(name+'process/','corr-'+traitChr[i]+'-'+traitChr[j],parms):
                continue
            DBWrite(np.array([]),name+'process/corr-'+traitChr[i]+'-'+traitChr[j],parms,toPickle=True)
            genCorrMatsHelp(traitChr[i],traitChr[j],parms)


def genCorrMatsHelp(trait_i,trait_j,parms): 
    name=parms['name']
    snpChr=parms['snpChr']
    traitData=DBRead(name+'process/traitData',toPickle=True)
    snpData=pd.read_csv(name+'process/snpData',toPickle=True)
    cpus=parms['smallCpu']
    K=float(len(snpData))
    
    if trait_i==trait_j:     
        mean=DBRead(name+'process/mean-'+trait_i,parms,toPickle=True).T
        std=DBRead(name+'process/std-'+trait_i,parms,toPickle=True).T
        
        traitData=traitData[traitData['chr']==trait_i]    
        corr=np.zeros([traitData.shape[0],traitData.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(cpus) as executor: 
            for snp in snpChr:
                if snp==trait_i:
                    continue
                
                futures.append(executor.submit(sameTrait,name,snp,trait_i,mean,std))

            for f in as_completed(futures):
                corr+=f.result()        
        
    else:
        mean_i=DBRead(name+'process/mean-'+trait_i,parms,toPickle=True).T
        std_i=DBRead(name+'process/std-'+trait_i,parms,toPickle=True).T

        mean_j=DBRead(name+'process/mean-'+trait_j,parms,toPickle=True).T
        std_j=DBRead(name+'process/std-'+trait_j,parms,toPickle=True).T

        traitData_i=traitData[traitData['chr']==trait_i]    
        traitData_j=traitData[traitData['chr']==trait_j]    

        corr=np.zeros([traitData_i.shape[0],traitData_j.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(cpus) as executor: 
            for snp in snpChr:
                if snp==trait_i or snp==trait_j:
                    continue

                futures.append(executor.submit(diffTrait,name,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j))
    
            for f in as_completed(futures):
                corr+=f.result()        

    print('writing corr mat '+trait_i+' - '+trait_j,flush=True)
    
    DBWrite((corr/K).astype('float16'),name+'process/corr-'+trait_i+'-'+trait_j,parms,toPickle=True)

def diffTrait(name,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j):
    df_i=DBRead(name+'process/z-'+snp+'-'+trait_i,parms,toPickle=True)
    df_j=DBRead(name+'process/z-'+snp+'-'+trait_j,parms,toPickle=True)

    df_i=(df_i-mean_i)/std_i
    df_j=(df_j-mean_j)/std_j

    return(np.matmul(df_i.T,df_j))
    
def sameTrait(scratchDir,snp,trait,mean,std):
    df=DBRead(name+'process/z-'+snp+'-'+trait,parms,toPickle=True)
    df=(df-mean)/std

    return(np.matmul(df.T,df))
    