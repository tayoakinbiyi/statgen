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
    
    DBSyncLocal(name+'process',parms)

    snpData=DBLocalRead(name+'process/snpData',parms)
    traitData=DBLocalRead(name+'process/traitData',parms)

    pre=('z-' if parms['wald'] else 'p-')
    
    for file in os.listdir(local+name+'score'):
        DBUpload(name+'score/'+file,parms)
    pdb.set_trace()
     
    for trait in traitChr:
        for snp in snpChr:
            if DBIsFile(name+'score',pre+snp+'-'+trait,parms):
                continue
            genScoresHelp(snp,trait,sum(snpData['chr']==snp),sum(traitData['chr']==trait),parms)
        
    return()

def genScoresHelp(snp,trait,numSnps,numTraits,parms):    
    numDecScore=parms['numDecScore']
    local=parms['local']
    name=parms['name']
    
    pre=('z-' if parms['wald'] else 'p-')
    preds=['-c',local+name+'process/preds.txt']
    pval='1' if parms['wald'] else '2'
    
    DBWrite(np.array([]),name+'score/'+pre+snp+'-'+trait,parms)
    
    print('starting ',snp,trait,flush=True)

    mat=np.empty([numSnps,numTraits])
    
    futures=[]
    #ans=gemma('chr1','chr1',1,preds,pval,local+name,parms['wald'])
    #pdb.set_trace()
    with ProcessPoolExecutor(parms['cpu']) as executor: 
        for k in range(numTraits):
            futures.append(executor.submit(gemma,snp,trait,k,preds,pval,local+name,parms['wald']))

        for f in as_completed(futures):
            ans=f.result()
            j=0
            k=ans[j];j+=1
            val=ans[j];j+=1
            snp=ans[j];j+=1
            trait=ans[j];j+=1

            if len(val)!=numSnps:
                print(snp,trait,k,'not shaped',len(val),numSnps,flush=True)
                sys.exit(1)
            
            mat[:,k]=val.astype('float16')

    print('writing ',snp,trait,flush=True)
    pdb.set_trace()
    DBWrite(mat,name+'score/'+pre+snp+'-'+trait,parms)
    
    return()
    
def gemma(snp,trait,k,preds,pval,path,wald):
    subprocess.run(['./gemma','-g',path+'process/geno-'+snp+'.txt','-p',path+'process/pheno-'+trait+'.txt',
        '-lmm',pval,'-o','z-'+snp+'-'+trait+'-'+str(k+1),'-k',path+'process/grm-'+snp+'.txt','-n',str(k+1),
        '-no-check','-silence','-notsnp']+preds) 

    df=pd.read_csv('output/z-'+snp+'-'+trait+'-'+str(k+1)+'.assoc.txt',sep='\t')
    os.remove('output/z-'+snp+'-'+trait+'-'+str(k+1)+'.assoc.txt')
    
    if wald:
        return(k,(df['beta']/df['se']).values.flatten(),snp,trait)
    else:
        return(k,df['p_lrt'].values.flatten(),snp,trait)