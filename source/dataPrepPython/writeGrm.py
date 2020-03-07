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
        cmd=[local+'ext/fastlmmc','-runGwasType','NORUN','-maxThreads',str(numCores),'-simOut','grm/fast-'+name,
             '-mpheno','1','-pheno','Y/Y.phe','-filesim','snps/'+name]
        
        subprocess.call(cmd)
        grmVal=pd.read_csv('grm/fast-'+name,sep='\t',header=0,index_col=0).values

    if 'gemma' in grmParm:
        cmd=[local+'ext/gemma','-o','gemma','-gk','2','-p','Y/Y.phe','-g','snps/'+name+'.bimbam']
        subprocess.call(cmd)     
        shutil.move('output/gemma.sXX.txt','grm/gemma-'+name)
        grmVal=pd.read_csv('grm/gemma-'+name,sep='\t',index_col=None,header=None).values

    if 'limix' in grmParm:
        grmVal=linear_kinship(np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float), verbose=False)
        
    if 'pylmm' in grmParm:
        bimBamFmt=np.loadtxt('snps/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].astype(float)
        grmVal = calculateKinship(bimBamFmt.T)

    if 'limNorm' in grmParm:
        grmVal=normalise_covariance(grmVal)
    
    np.savetxt('grm/gemma-'+name,grmVal,delimiter='\t')
    cmd=[local+'ext/gemma','-k','grm/gemma-'+name,'-eigen','-o','gemma','-g','snps/'+name+'.bimbam','-p','Y/Y.phe']
    subprocess.call(cmd)
    subprocess.call(['mkdir','-p','grm/gemma-eigen-'+name])
    subprocess.call(['mv','output/gemma.eigenU.txt','grm/gemma-eigen-'+name+'/U'])
    subprocess.call(['mv','output/gemma.eigenD.txt','grm/gemma-eigen-'+name+'/D'])

    np.savetxt('grm/limix-'+name,grmVal,delimiter='\t')
    
    fastGrm=pd.DataFrame(grmVal)
    fastGrm.index=['0 '+str(x) for x in np.arange(numSubjects)]
    fastGrm.columns=['0 '+str(x) for x in np.arange(numSubjects)]
    fastGrm.index.name='var'
    fastGrm.to_csv('grm/fast-'+name,sep='\t',header=True,index=True)  
    cmd=[local+'ext/fastlmmc','-mpheno','1','-maxThreads',str(numCores),'-file','snps/'+name,
         '-sim','grm/fast-'+name,'-eigenOut','grm/fast-eigen-'+name,'-pheno','Y/Y.phe']
    subprocess.call(cmd)
    
    Kva,Kve = np.linalg.eigh(grmVal)
    subprocess.call(['mkdir','-p','grm/pylmm-eigen-'+name])
    np.savetxt('grm/pylmm-eigen-'+name+'/K',grmVal,delimiter='\t')
    np.savetxt('grm/pylmm-eigen-'+name+'/Kva',Kva,delimiter='\t')
    np.savetxt('grm/pylmm-eigen-'+name+'/Kve',Kve,delimiter='\t' )
    
    np.savetxt('grm/Lgrm-'+name,makePSD(grmVal,corr=False),delimiter='\t')        

    return()
