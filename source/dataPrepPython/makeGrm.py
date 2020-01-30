import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random

from opPython.DB import *
from genPython.makePSD import *

def makeGrm(parms,snp,makeL):
    local=parms['local']
    numCores=parms['numCores']
    grm=parms['grm']

    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpIds=snpData['ID'][snpData['chr']!=snp].values.flatten()
    
    if len(snpIds)==0:
        return()
        
    pd.DataFrame({'Family ID':snpIds,'Individual ID':0}).to_csv('ped/extract',sep='\t',index=False,header=False)
                
    if grm=='fast':
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-runGwasType','RUN','-extractSim','ped/extract',
             '-pheno','ped/snp.fam','-maxThreads',str(numCores),'-simOut','grm/fast-'+str(snp),
             '-eigenOut','grm/eigen-'+str(snp),'-mpheno','1']
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+str(snp),sep='\t',header=0,index_col=0)
        N=len(grmVal)
        np.savetxt('grm/gemma-'+str(snp),grmVal.values,delimiter='\t')
        os.symlink('fast-'+str(snp),'grm/grm-'+str(snp))
    else:
        cmd=[local+'ext/gemma','-bfile','ped/snp','-o','gemma','-gk',str(grm),'-snps','ped/extract']
        subprocess.call(cmd)     
        nm='c' if grm==1 else 's'
        shutil.move('output/gemma.'+nm+'XX.txt','grm/gemma-'+str(snp))
        grmVal=pd.read_csv('grm/gemma-'+str(snp),sep='\t',index_col=None,header=None)
        N=len(grmVal)
        mouseIds=np.arange(N)
        grmVal.index=[str(x)+' 0' for x in mouseIds]
        grmVal.columns=[str(x)+' 0' for x in mouseIds]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+str(snp),sep='\t',header=True,index=True)       
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-pheno','ped/snp.fam','-mpheno','1','-maxThreads',str(numCores),
             '-sim','grm/fast-'+str(snp),'-eigenOut','grm/eigen-'+str(snp)]
        subprocess.call(cmd)
        os.symlink('gemma-'+str(snp),'grm/grm-'+str(snp))        
        
    if makeL:
        np.savetxt('LZCorr/Lgrm-'+str(snp),makePSD(grmVal.values),delimiter='\t')        

    return()
