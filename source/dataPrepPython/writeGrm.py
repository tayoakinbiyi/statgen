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
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from pylmm.lmm import calculateKinship

def writeGrm(parms,name,decomp=True):
    local=parms['local']
    numCores=parms['numCores']
    grmParm=parms['grmParm']
    numSubjects=parms['parms'][1]
    
    name=str(name)
    
    if 'fast' in grmParm:
        cmd=[local+'ext/fastlmmc','-runGwasType','NORUN','-maxThreads',str(numCores),'-simOut','grm/grmWithIndex-'+name,
             '-mpheno','1','-pheno','Y/Y.phe','-filesim','snps/'+name]
        
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/grmWithIndex-'+name,sep='\t',header=0,index_col=0).values

    if 'gemma' in grmParm:
        cmd=[local+'ext/gemma','-o','gemma','-gk','2','-p','Y/Y.phe','-g','snps/'+name+'.bimbam']
        subprocess.call(cmd)     
        grmVal=np.loadtxt('output/gemma.sXX.txt',delimiter='\t')

    if 'limix' in grmParm:
        grmVal=linear_kinship(np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float), verbose=False)
        
    if 'limNorm' in grmParm:
        grmVal=linear_kinship(np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float), verbose=False)
        grmVal=normalise_covariance(grmVal)
        
    if 'pylmm' in grmParm:
        bimBamFmt=np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].astype(float)
        grmVal = calculateKinship(bimBamFmt.T)
        
    if 'manual' in grmParm:
        bimBamFmt=np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].astype(float)
        X=(bimBamFmt-np.mean(bimBamFmt,axis=1).reshape(-1,1))/np.std(bimBamFmt,axis=1).reshape(-1,1)
        grmVal=np.matmul(X.T,X)/X.shape[0]
        
    if 'corr' in grmParm:
        grmVal=np.corrcoef(np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].astype(float),rowvar=False)
    
    np.savetxt('grm/grm-'+name,grmVal,delimiter='\t')
    grmWithIndex=pd.DataFrame(grmVal)
    grmWithIndex.index=['0 '+str(x) for x in np.arange(numSubjects)]
    grmWithIndex.columns=['0 '+str(x) for x in np.arange(numSubjects)]
    grmWithIndex.index.name='var'
    grmWithIndex.to_csv('grm/grmWithIndex-'+name,sep='\t',header=True,index=True)  

    cmd=[local+'ext/gemma','-k','grm/grm-'+name,'-eigen','-o','gemma','-g','snps/'+name+'.bimbam','-p','Y/Y.phe']
    subprocess.call(cmd)
    subprocess.call(['mkdir','-p','grm/gemma-'+name])
    subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-'+name+'/U'])
    subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-'+name+'/D'])
    
    cmd=[local+'ext/fastlmmc','-mpheno','1','-maxThreads',str(numCores),'-file','snps/'+name,
         '-sim','grm/grmWithIndex-'+name,'-eigenOut','grm/fast-'+name,'-pheno','Y/Y.phe']
    subprocess.call(cmd)
    
    Kva,Kve = np.linalg.eigh(grmVal)
    subprocess.call(['mkdir','-p','grm/eigh-'+name])
    np.savetxt('grm/eigh-'+name+'/Kva',Kva,delimiter='\t')
    np.savetxt('grm/eigh-'+name+'/Kve',Kve,delimiter='\t' )
    
    np.savetxt('grm/Lgrm-'+name,makePSD(grmVal,corr=False),delimiter='\t')        

    return()
