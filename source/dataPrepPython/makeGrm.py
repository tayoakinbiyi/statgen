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

    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpIds=snpData['ID'][snpData['chr'].isin(snpSet)].values.flatten()
    
    if len(snpIds)==0:
        return()
        
    pd.DataFrame({'Family ID':snpIds,'Individual ID':0}).to_csv('ped/extract',sep='\t',index=False,header=False)
             
    if grm=='fast':
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-runGwasType','RUN','-extractSim','ped/extract',
             '-pheno','ped/snp.fam','-maxThreads',str(numCores),'-simOut','grm/fast-'+str(name),'-mpheno','1']
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+str(name),sep='\t',header=0,index_col=0)
        N=len(grmVal)
        np.savetxt('grm/gemma-'+str(name),grmVal.values,delimiter='\t')
    else:
        if grm=='gemmaStd':
            op=2
            nm='s'
        else:
            op=1
            nm='c'

        cmd=[local+'ext/gemma','-bfile','ped/snp','-o','gemma','-gk',str(op),'-snps','ped/extract']
        subprocess.call(cmd)     
        shutil.move('output/gemma.'+nm+'XX.txt','grm/gemma-'+str(name))
        grmVal=pd.read_csv('grm/gemma-'+str(name),sep='\t',index_col=None,header=None)
        N=len(grmVal)
        mouseIds=np.arange(N)
        grmVal.index=[str(x)+' 0' for x in mouseIds]
        grmVal.columns=[str(x)+' 0' for x in mouseIds]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+str(name),sep='\t',header=True,index=True)       
        
    cmd=[local+'ext/fastlmmc','-bfile','ped/snp','-pheno','ped/snp.fam','-mpheno','1','-maxThreads',str(numCores),
             '-sim','grm/fast-'+str(name),'-eigenOut','grm/fast-eigen-'+str(name)]
    subprocess.call(cmd)
    cmd=[local+'ext/gemma','-bfile','ped/snp','-k','grm/gemma-'+str(name),'-eigen','-o','gemma']
    subprocess.call(cmd)
    subprocess.call(['mkdir','-p','grm/gemma-eigen-'+str(name)])
    subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-eigen-'+str(name)+'/U'])
    subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-eigen-'+str(name)+'/D'])
        
    np.savetxt('LZCorr/Lgrm-'+str(name),makePSD(grmVal.values),delimiter='\t')        

    return()
