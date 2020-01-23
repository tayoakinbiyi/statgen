import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random

from opPython.DB import *
from genPython.makePSD import *

def makeGrm(parms,snp,fastGrm,makeL):
    local=parms['local']
    numCores=parms['numCores']

    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpIds=snpData['ID'][snpData['chr']!=snp].values.flatten()
    
    if len(snpIds)==0:
        return()
        
    pd.DataFrame({'Family ID':snpIds,'Individual ID':0}).to_csv('ped/extract',sep='\t',index=False,header=False)
                
    if fastGrm:
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-runGwasType','RUN','-extractSim','ped/extract',
             '-pheno','ped/snp.fam','-maxThreads',str(numCores),'-simOut','grm/fast-'+str(snp),
             '-eigenOut','grm/eigen-'+str(snp),'-mpheno','1']
        subprocess.call(cmd)
        grm=pd.read_csv('grm/fast-'+str(snp),sep='\t',header=0,index_col=0)
        N=len(grm)
        np.savetxt('grm/gemma-'+str(snp),grm.values,delimiter='\t')
        os.symlink('fast-'+str(snp),'grm/grm-'+str(snp))
    else:
        cmd=[local+'ext/gemma','-bfile','ped/snp','-o','gemma','-gk','1','-snps','ped/extract']
        subprocess.call(cmd)     
        shutil.move('output/gemma.cXX.txt','grm/gemma-'+str(snp))
        grm=pd.read_csv('grm/gemma-'+str(snp),sep='\t',index_col=None,header=None)
        N=len(grm)
        mouseIds=np.arange(N)
        grm.index=[str(x)+' 0' for x in mouseIds]
        grm.columns=[str(x)+' 0' for x in mouseIds]
        grm.index.name='var'
        
        grm.to_csv('grm/fast-'+str(snp),sep='\t',header=True,index=True)       
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-pheno','ped/snp.fam','-mpheno','1','-maxThreads',str(numCores),
             '-sim','grm/fast-'+str(snp),'-eigenOut','grm/eigen-'+str(snp)]
        subprocess.call(cmd)
        os.symlink('gemma-'+str(snp),'grm/grm-'+str(snp))        
        
    if makeL:
        np.savetxt('LZCorr/Lgrm-'+str(snp),makePSD(grm.values),delimiter='\t')        

    return()
