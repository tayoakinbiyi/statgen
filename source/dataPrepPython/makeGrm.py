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
    grm=parms['grm']

    snpData=pd.read_csv('inputs/snpData',index_col=None,header=0,sep='\t')
    snpIds=snpData['ID'][snpData['chr'].isin(snpSet)].values.flatten()
    
    name=str(name)
    
    if len(snpIds)==0:
        return()
                   
    if 'fast' in grm:
        cmd=[local+'ext/fastlmmc','-runGwasType','NORUN','-maxThreads',str(numCores),'-simOut','grm/fast-'+name,
             '-mpheno','1','-pheno','inputs/Y.phe','-filesim','inputs/'+name]
        
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+name,sep='\t',header=0,index_col=0)
        N=len(grmVal)
        np.savetxt('grm/gemma-'+name,grmVal.values,delimiter='\t')
    if 'gemma' in grm:
        if 'std' in grm:
            op=2
            nm='s'
        if 'central' in grm:
            op=1
            nm='c'
        
        cmd=[local+'ext/gemma','-o','gemma','-gk',str(op),'-p','inputs/Y.phe','-g','inputs/'+name+'.bimbam']
        subprocess.call(cmd)     
        shutil.move('output/gemma.'+nm+'XX.txt','grm/gemma-'+name)
        grmVal=pd.read_csv('grm/gemma-'+name,sep='\t',index_col=None,header=None)
        N=len(grmVal)
        grmVal.index=['0 '+str(x) for x in np.arange(N)]
        grmVal.columns=['0 '+str(x) for x in np.arange(N)]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+name,sep='\t',header=True,index=True)       
    
    cmd=[local+'ext/fastlmmc','-mpheno','1','-maxThreads',str(numCores),'-file','inputs/'+name,
         '-sim','grm/fast-'+name,'-eigenOut','grm/fast-eigen-'+name,'-pheno','inputs/Y.phe']
    subprocess.call(cmd)
    cmd=[local+'ext/gemma','-k','grm/gemma-'+name,'-eigen','-o','gemma','-g','inputs/'+name+'.bimbam','-p','inputs/Y.phe']
    subprocess.call(cmd)
    subprocess.call(['mkdir','-p','grm/gemma-eigen-'+name])
    subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-eigen-'+name+'/U'])
    subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-eigen-'+name+'/D'])
        
    np.savetxt('LZCorr/Lgrm-'+name,makePSD(grmVal.values),delimiter='\t')        

    return()