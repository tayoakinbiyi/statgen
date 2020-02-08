import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from scipy.stats import chi2,t,norm

from opPython.DB import *
from opPython.verboseArrCheck import *

def genZScores(parms):
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    simLearnType=parms['simLearnType']
    fastlmm=parms['fastlmm']
    N=parms['numSubjects']
    
    DBLog('genZScores')
    DBCreateFolder('output',parms)
    
    traitData=pd.read_csv('ped/traitData',index_col=None,header=0,sep='\t')
    traitData=traitData[traitData['chr'].isin(traitChr)]
    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpData=snpData[snpData['chr'].isin(snpChr)]
    
    fileOp='fast-' if fastlmm else 'gemma-'
    
    for trait in traitChr:
        for snp in snpChr:
            nameParm=str(snp)+'-'+str(trait)

            numTraits=sum(traitData['chr']==trait)
            numSnps=sum(snpData['chr']==snp)
            
            pLRT=np.full([numSnps,numTraits],np.nan)
            pWald=np.full([numSnps,numTraits],np.nan)
            waldStat=np.full([numSnps,numTraits],np.nan)
            AltLogLike=np.full([numSnps,numTraits],np.nan)
            beta=np.full([numSnps,numTraits],np.nan)
            se=np.full([numSnps,numTraits],np.nan)
                        
            with ProcessPoolExecutor(numCores) as executor:
                futures=[]
                    
                for core in range(numCores):
                    traitRange=np.arange(core*int(np.ceil(numTraits/numCores)),min(numTraits,(core+1)*int(
                        np.ceil(numTraits/numCores))))
                    if len(traitRange)==0:
                        continue

                    futures+=[executor.submit(genZScoresHelp,str(core),str(snp),str(trait),traitRange,parms,fastlmm,N)]

                for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                    ans=f.result()
                    traitRange=ans['traitRange']
                    
                    waldStat[:,traitRange]=ans['waldStat']
                    pLRT[:,traitRange]=ans['pLRT']
                    pWald[:,traitRange]=ans['pWald']
                    AltLogLike[:,traitRange]=ans['AltLogLike']
                    beta[:,traitRange]=ans['beta']
                    se[:,traitRange]=ans['se']

            np.savetxt('score/'+fileOp+'waldStat-'+nameParm,waldStat,delimiter='\t')
            np.savetxt('score/'+fileOp+'pLRT-'+nameParm,pLRT,delimiter='\t')
            np.savetxt('score/'+fileOp+'pWald-'+nameParm,pWald,delimiter='\t') 
            np.savetxt('score/'+fileOp+'AltLogLike-'+nameParm,AltLogLike,delimiter='\t') 
            np.savetxt('score/'+fileOp+'beta-'+nameParm,beta,delimiter='\t') 
            np.savetxt('score/'+fileOp+'se-'+nameParm,se,delimiter='\t') 
            
            for file in ['waldStat','pLRT','pWald']:
                if os.path.exists('score/'+file+'-'+nameParm):
                    os.remove('score/'+file+'-'+nameParm)
                os.symlink(fileOp+file+'-'+nameParm,'score/'+file+'-'+nameParm)
        
    return()

def genZScoresHelp(core,snp,trait,traitRange,parms,fastlmm,N):    
    if fastlmm:
        return(runFastlmm(core,snp,trait,traitRange,parms,N))
    else:
        return(runGemma(core,snp,trait,traitRange,parms,N))        

def runFastlmm(core,snp,trait,traitRange,parms,N):
    simLearnType=parms['simLearnType']
    local=parms['local']
    
    nameParm=snp+'-'+trait+'-'+core
    
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]

    for traitInd in traitRange:
        cmd=[local+'ext/fastlmmc','-bfile','ped/snp-'+snp,'-covar','ped/cov.phe','-pheno','ped/Y-'+trait+'.phe',
             '-eigen','grm/fast-eigen-'+snp,'-mpheno',str(traitInd+1),'-out','output/fastlmm-'+nameParm, '-maxThreads','1',
             '-simLearnType',simLearnType,'-ML','-Ftest']

        subprocess.call(cmd)

        df=pd.read_csv('output/fastlmm-'+nameParm,header=0,index_col=None,sep='\t')

        df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
        df=df.sort_values(by='SNP')
    
        waldStat+=[(df['SNPWeight']/df['SNPWeightSE']).values.reshape(-1,1)]
        pLRT+=[chi2.sf(2*(df['AltLogLike']-df['NullLogLike']),1).reshape(-1,1)]
        pWald+=[df['Pvalue'].values.reshape(-1,1)]
        AltLogLike+=[df['AltLogLike'].values.reshape(-1,1)]
        beta+=[df['SNPWeight'].values.reshape(-1,1)]
        se+=[df['SNPWeightSE'].values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    pLRT=np.concatenate(pLRT,axis=1)
    pWald=np.concatenate(pWald,axis=1)
    AltLogLike=np.concatenate(AltLogLike,axis=1)
    beta=np.concatenate(beta,axis=1)
    se=np.concatenate(se,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'pLRT':pLRT,
            'pWald':pWald,
            'AltLogLike':AltLogLike,
            'beta':beta,
            'se':se
           })
                                  
def runGemma(core,snp,trait,traitRange,parms,N):
    local=parms['local']
    
    nameParm=snp+'-'+trait+'-'+core
    
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    # '-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U',
    for traitInd in traitRange:
        cmd=[local+'ext/gemma','-bfile','ped/snp-'+snp,'-lm','4','-o','gemma-'+nameParm,
             '-n',str(traitInd+1),'-c','ped/cov.txt',
             '-p','ped/Y-'+trait+'.txt']

        subprocess.run(cmd) 

        df=pd.read_csv('output/gemma-'+nameParm+'.assoc.txt',header=0,index_col=None,sep='\t')

        waldStat+=[(df['beta']/df['se']).values.reshape(-1,1)]
        pLRT+=[df['p_lrt'].values.reshape(-1,1)]
        pWald+=[df['p_wald'].values.reshape(-1,1)]
        AltLogLike+=[df['p_wald'].values.reshape(-1,1)]#logl_H1
        beta+=[df['beta'].values.reshape(-1,1)]
        se+=[df['se'].values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    pLRT=np.concatenate(pLRT,axis=1)
    pWald=np.concatenate(pWald,axis=1)
    AltLogLike=np.concatenate(AltLogLike,axis=1)
    beta=np.concatenate(beta,axis=1)
    se=np.concatenate(se,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'pLRT':pLRT,
            'pWald':pWald,
            'AltLogLike':AltLogLike,
            'beta':beta,
            'se':se
           })