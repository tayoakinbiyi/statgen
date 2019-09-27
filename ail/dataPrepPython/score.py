import pandas as pd
import numpy as np
import subprocess
import pdb
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed

from ail.opPython.DB import *

def score(parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    wald=parms['wald']
    
    DBSyncLocal(name+'process',parms)

    snpData=DBRead(name+'process/snpData',parms,toPickle=True)
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    for trait in traitChr:
        for snp in snpChr:
            if DBIsFile(name+'score','p-'+snp+'-'+trait,parms):
                print('found ','p-'+snp+'-'+trait,flush=True)
                continue
            
            genScoresHelp(snp,trait,sum(snpData['chr']==snp),sum(traitData['chr']==trait),parms)
        
    return()

def genScoresHelp(snp,trait,numSnps,numTraits,parms):    
    numDecScore=parms['numDecScore']
    local=parms['local']
    name=parms['name']
    wald=parms['wald']
        
    DBWrite(np.array([]),name+'score/p-'+snp+'-'+trait,parms,toPickle=True)
    
    print('starting ',snp,trait,flush=True)

    mat=np.empty([numSnps,numTraits])
    
    futures=[]
    with ProcessPoolExecutor(parms['cpu']) as executor: 
        for k in range(numTraits):
            futures.append(executor.submit(gemma,snp,trait,k,local,name,wald))

        for f in as_completed(futures):
            ans=f.result()
            j=0
            k=ans[j];j+=1
            val=ans[j];j+=1
            snp=ans[j];j+=1
            trait=ans[j];j+=1
            
            mat[:,k]=val

    print('writing ',snp,trait,flush=True)
    
    DBWrite(mat,name+'score/p-'+snp+'-'+trait,parms,toPickle=True)
    
    return()
    
def gemma(snp,trait,k,local,name,wald):
    path=local+name
    pval='1' if wald else '2'
    
    cmd=[local+'ext/gemma','-g',path+'process/geno-'+snp+'.txt','-p',path+'process/pheno-'+trait+'.txt',
        '-lmm',pval,'-o',name[:-1]+'-'+snp+'-'+trait+'-'+str(k+1),'-k',path+'process/grm-'+snp+'.txt','-n',str(k+1),
        '-c',path+'process/preds.txt']
    subprocess.run(cmd) 

    df=pd.read_csv('output/'+name[:-1]+'-'+snp+'-'+trait+'-'+str(k+1)+'.assoc.txt',sep='\t')
    os.remove('output/'+name[:-1]+'-'+snp+'-'+trait+'-'+str(k+1)+'.assoc.txt')
    os.remove('output/'+name[:-1]+'-'+snp+'-'+trait+'-'+str(k+1)+'.log.txt')
    
    if wald:
        val=(df['beta']/df['se']).values.flatten()
    else:
        val=df['p_lrt'].values.flatten()
    
    return(k,val,snp,trait,df['rs'])