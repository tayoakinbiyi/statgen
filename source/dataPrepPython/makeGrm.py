import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random

from opPython.DB import *
from genPython.makePSD import *

def makeGrm(parms,name,snpSet):
    local=parms['local']
    numCores=parms['numCores']
    data=parms['data']

    snpData=pd.read_csv('inputs/snpData',index_col=None,header=0,sep='\t')
    snpIds=snpData['ID'][snpData['chr'].isin(snpSet)].values.flatten()
    
    name=str(name)
    
    if len(snpIds)==0:
        return()
            
    if 'fastGrm' in data:
        cmd=[local+'ext/fastlmmc','-bfile','inputs/'+name,'-runGwasType','RUN',
             '-maxThreads',str(numCores),'-simOut','grm/fast-'+name,'-mpheno','1','-pheno','inputs/Y.phe']
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+name,sep='\t',header=0,index_col=0)
        N=len(grmVal)
        np.savetxt('grm/gemma-'+name,grmVal.values,delimiter='\t')
    else:
        if 'gemmaStdGrm' in data:
            op=2
            nm='s'
        else:
            op=1
            nm='c'
        
        cmd=[local+'ext/gemma','-bfile','inputs/'+name,'-o','gemma','-gk',str(op)]
        subprocess.call(cmd)     
        shutil.move('output/gemma.'+nm+'XX.txt','grm/gemma-'+name)
        grmVal=pd.read_csv('grm/gemma-'+name,sep='\t',index_col=None,header=None)
        N=len(grmVal)
        grmVal.index=['0 '+str(x) for x in np.arange(N)]
        grmVal.columns=['0 '+str(x) for x in np.arange(N)]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+name,sep='\t',header=True,index=True)       
        
    cmd=[local+'ext/fastlmmc','-bfile','inputs/'+name,'-mpheno','1','-maxThreads',str(numCores),'-runGwasType','RUN', 
         '-sim','grm/fast-'+name,'-eigenOut','grm/fast-eigen-'+name,'-pheno','inputs/Y.phe']
    subprocess.call(cmd)
    cmd=[local+'ext/gemma','-bfile','inputs/'+name,'-k','grm/gemma-'+name,'-eigen','-o','gemma']
    subprocess.call(cmd)
    subprocess.call(['mkdir','-p','grm/gemma-eigen-'+name])
    subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-eigen-'+name+'/U'])
    subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-eigen-'+name+'/D'])
        
    np.savetxt('LZCorr/Lgrm-'+name,makePSD(grmVal.values),delimiter='\t')        

    return()
