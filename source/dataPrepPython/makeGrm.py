import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random

from opPython.DB import *
from genPython.makePSD import *
from limix.qc import normalise_covariance
from limix.stats import linear_kinship

def makeGrm(parms,name,decomp=True):
    local=parms['local']
    numCores=parms['numCores']
    grm=parms['grm']
    numSubjects=parms['parms'][1]
    
    name=str(name)
    
    if 'fast' in grm:
        cmd=[local+'ext/fastlmmc','-runGwasType','NORUN','-maxThreads',str(numCores),'-simOut','grm/fast-'+name,
             '-mpheno','1','-pheno','inputs/Y.phe','-filesim','inputs/'+name]
        
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+name,sep='\t',header=0,index_col=0)
        N=len(grmVal)
        np.savetxt('grm/gemma-'+name,grmVal.values,delimiter='\t')
        np.savetxt('grm/limix-'+name,grmVal.values,delimiter='\t')
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
        np.savetxt('grm/limix-'+name,grmVal,delimiter='\t')
        grmVal.index=['0 '+str(x) for x in np.arange(numSubjects)]
        grmVal.columns=['0 '+str(x) for x in np.arange(numSubjects)]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+name,sep='\t',header=True,index=True)  
    if 'limix' in grm:
        grm=linear_kinship(np.loadtxt('inputs/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float), verbose=False)
        np.savetxt('grm/gemma-'+name,grm,delimiter='\t')
        np.savetxt('grm/limix-'+name,grm,delimiter='\t')
        grmVal=pd.DataFrame(grm)
        
        grmVal.index=['0 '+str(x) for x in np.arange(numSubjects)]
        grmVal.columns=['0 '+str(x) for x in np.arange(numSubjects)]
        grmVal.index.name='var'
        
        grmVal.to_csv('grm/fast-'+name,sep='\t',header=True,index=True)  
    
    if 'gcta' in grm:
        cmd=[local+'ext/gcta64','--bfile','inputs/'+name,'--make-grm','--out','grm/gcta-'+name]   
        subprocess.call(cmd)
        pdb.set_trace()

    if 'limNorm' in grm:
        np.savetxt('grm/gemma-'+name,normalise_covariance(np.loadtxt('grm/gemma-'+name,delimite='\t')),delimiter='\t')
        np.savetxt('grm/fast-'+name,normalise_covariance(np.loadtxt('grm/fast-'+name,delimite='\t',
            dtype='str')[1:,1:].astype(float)),delimiter='\t')
        
    if decomp:
        cmd=[local+'ext/fastlmmc','-mpheno','1','-maxThreads',str(numCores),'-file','inputs/'+name,
             '-sim','grm/fast-'+name,'-eigenOut','grm/fast-eigen-'+name,'-pheno','inputs/Y.phe']
        subprocess.call(cmd)
        cmd=[local+'ext/gemma','-k','grm/gemma-'+name,'-eigen','-o','gemma','-g','inputs/'+name+'.bimbam','-p','inputs/Y.phe']
        subprocess.call(cmd)
        subprocess.call(['mkdir','-p','grm/gemma-eigen-'+name])
        subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-eigen-'+name+'/U'])
        subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-eigen-'+name+'/D'])
    
    np.savetxt('LZCorr/Lgrm-'+name,makePSD(grmVal.values,corr=False),delimiter='\t')        

    return()
