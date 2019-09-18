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
            if DBIsFile(name+'corr','corr-'+traitChr[i]+'-'+traitChr[j],parms):
                continue
            DBWrite(np.array([]),name+'corr/corr-'+traitChr[i]+'-'+traitChr[j],parms,toPickle=True)
            genCorrMatsHelp(traitChr[i],traitChr[j],parms)
    
    return()


def genCorrMatsHelp(trait_i,trait_j,parms): 
    name=parms['name']
    snpChr=parms['snpChr']
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)
    snpData=DBRead(name+'process/snpData',parms,toPickle=True)
    cpus=parms['smallCpu']
    K=float(len(snpData))
    
    if trait_i==trait_j:     
        mean=DBRead(name+'corr/mean-'+trait_i,parms,toPickle=True)
        std=DBRead(name+'corr/std-'+trait_i,parms,toPickle=True)
        
        traitData=traitData[traitData['chr']==trait_i]    
        corr=np.zeros([traitData.shape[0],traitData.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(cpus) as executor: 
            for snp in snpChr:
                if snp==trait_i:
                    continue
                
                futures.append(executor.submit(sameTrait,parms,snp,trait_i,mean,std))

            for f in as_completed(futures):
                corr+=f.result()        
        
    else:
        mean_i=DBRead(name+'corr/mean-'+trait_i,parms,toPickle=True)
        std_i=DBRead(name+'corr/std-'+trait_i,parms,toPickle=True)

        mean_j=DBRead(name+'corr/mean-'+trait_j,parms,toPickle=True)
        std_j=DBRead(name+'corr/std-'+trait_j,parms,toPickle=True)

        traitData_i=traitData[traitData['chr']==trait_i]    
        traitData_j=traitData[traitData['chr']==trait_j]    

        corr=np.zeros([traitData_i.shape[0],traitData_j.shape[0]])
        
        futures=[]
        with ProcessPoolExecutor(cpus) as executor: 
            for snp in snpChr:
                if snp==trait_i or snp==trait_j:
                    continue

                futures.append(executor.submit(diffTrait,parms,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j))
    
            for f in as_completed(futures):
                corr+=f.result()        

    print('writing corr mat '+trait_i+' - '+trait_j,flush=True)
    
    DBWrite((corr/K).astype('float16'),name+'corr/corr-'+trait_i+'-'+trait_j,parms,toPickle=True)

def diffTrait(parms,snp,trait_i,trait_j,mean_i,mean_j,std_i,std_j):
    name=parms['name']
    
    df_i=DBRead(name+'score/p-'+snp+'-'+trait_i,parms,toPickle=True)
    df_j=DBRead(name+'score/p-'+snp+'-'+trait_j,parms,toPickle=True)

    df_i=(df_i-mean_i)/std_i
    df_j=(df_j-mean_j)/std_j

    return(np.matmul(df_i.T,df_j))
    
def sameTrait(parms,snp,trait,mean,std):
    name=parms['name']
    
    df=DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True)
    df=(df-mean)/std

    return(np.matmul(df.T,df))
    