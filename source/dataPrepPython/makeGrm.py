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
        grmVal=pd.read_csv('grm/fast-'+name,sep='\t',header=0,index_col=0).values

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
        grmVal=pd.read_csv('grm/gemma-'+name,sep='\t',index_col=None,header=None).values

    if 'limix' in grm:
        grmVal=linear_kinship(np.loadtxt('inputs/'+name+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float), verbose=False)
    
    if 'gcta' in grm:
        subprocess.call(['mkdir','-p','grm/gcta-'+name])
        cmd=[local+'ext/gcta64','--bfile','inputs/'+name,'--make-grm','--out','grm/gcta-'+name+'/grm','--threads',str(numCores)]   
        subprocess.call(cmd)

        gctaString='''
        ReadGRMBin=function(){
          prefix=\'grm/gcta-'''+name+'''/grm\'
          AllN=F
          size=4
          sum_i=function(i){
            return(sum(1:i))
          }
          BinFileName=paste(prefix,".grm.bin",sep="")
          NFileName=paste(prefix,".grm.N.bin",sep="")
          IDFileName=paste(prefix,".grm.id",sep="")
          id = read.table(IDFileName)
          n=dim(id)[1]
          BinFile=file(BinFileName, "rb");
          grm=readBin(BinFile, n=n*(n+1)/2, what=numeric(0), size=size)
          NFile=file(NFileName, "rb");
          if(AllN==T){
            N=readBin(NFile, n=n*(n+1)/2, what=numeric(0), size=size)
          }
          else N=readBin(NFile, n=1, what=numeric(0), size=size)
          i=sapply(1:n, sum_i)
          return(list(diag=grm[i], off=grm[-i], id=id, N=N))
        }
        '''
        gctaF=SignatureTranslatedAnonymousPackage(gctaString,'ReadGRMBin')    
        grmValList=gctaF.ReadGRMBin()
        grmVal=np.diag(grmValList[0])
        grmVal[np.triu_indices(numSubjects,1)]=grmValList[1]
        grmVal[np.tril_indices(numSubjects,-1)]=grmValList[1][::-1]

    if 'limNorm' in grm:
        grmVal=normalise_covariance(grmVal)
    
    np.savetxt('grm/gemma-'+name,grmVal,delimiter='\t')
    cmd=[local+'ext/gemma','-k','grm/gemma-'+name,'-eigen','-o','gemma','-g','inputs/'+name+'.bimbam','-p','inputs/Y.phe']
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
    cmd=[local+'ext/fastlmmc','-mpheno','1','-maxThreads',str(numCores),'-file','inputs/'+name,
         '-sim','grm/fast-'+name,'-eigenOut','grm/fast-eigen-'+name,'-pheno','inputs/Y.phe']
    subprocess.call(cmd)
    
    np.savetxt('LZCorr/Lgrm-'+name,makePSD(grmVal,corr=False),delimiter='\t')        

    return()
