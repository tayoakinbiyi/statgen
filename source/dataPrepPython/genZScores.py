import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from scipy.stats import chi2

from opPython.DB import *
from opPython.verboseArrCheck import *

def genZScores(parms):
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    simLearnType=parms['simLearnType']
    fastlmm=parms['fastlmm']
    
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

            if os.path.exists('holds/genZScores-'+nameParm):
                continue

            np.savetxt('holds/genZScores-'+nameParm,np.array([]),delimiter='\t')

            numTraits=sum(traitData['chr']==trait)
            numSnps=sum(snpData['chr']==snp)
            
            pLRT=np.full([numSnps,numTraits],np.nan)
            pWald=np.full([numSnps,numTraits],np.nan)
            waldStat=np.full([numSnps,numTraits],np.nan)
            AltLogLike=np.full([numSnps,numTraits],np.nan)
            beta=np.full([numSnps,numTraits],np.nan)
            se=np.full([numSnps,numTraits],np.nan)
                        
            with ProcessPoolExecutor(numCores) as executor:
                traitInd=0
                while traitInd<numTraits:
                    futures=[]
                    
                    for core in range(min(numTraits-traitInd,numCores)):
                        futures+=[executor.submit(genZScoresHelp,str(core),str(snp),str(trait),traitInd,parms,fastlmm)]
                        traitInd+=1
                    
                    for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                        ans=f.result()
                        waldStat[:,ans['traitInd']]=ans['waldStat']
                        pLRT[:,ans['traitInd']]=ans['pLRT']
                        pWald[:,ans['traitInd']]=ans['pWald']
                        AltLogLike[:,ans['traitInd']]=ans['AltLogLike']
                        beta[:,ans['traitInd']]=ans['beta']
                        se[:,ans['traitInd']]=ans['se']
                        
                        DBLog('genZScores '+nameParm+'-'+str(ans['traitInd']+1)+' of '+str(numTraits)+
                              '\n waldStat '+verboseArrCheck(ans['waldStat'])+'\n pLRT '+verboseArrCheck(ans['pLRT'])+
                              '\n pWald '+verboseArrCheck(ans['pWald']))

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

def genZScoresHelp(core,snp,trait,traitInd,parms,fastlmm):    
    if fastlmm:
        return(runFastlmm(core,snp,trait,traitInd,parms))
    else:
        return(runGemma(core,snp,trait,traitInd,parms))        

def runFastlmm(core,snp,trait,traitInd,parms):
    simLearnType=parms['simLearnType']
    local=parms['local']
    
    nameParm=snp+'-'+trait+'-'+core
    cmd=[local+'ext/fastlmmc','-bfile','ped/snp-'+snp,'-covar','ped/cov.phe','-pheno','ped/Y-'+trait+'.phe',
         '-eigen','grm/eigen-'+snp,'-mpheno',str(traitInd+1),'-out','output/fastlmm-'+nameParm, '-maxThreads','1',
         '-simLearnType',simLearnType,'-ML','-Ftest']
    
    subprocess.call(cmd)
    
    df=pd.read_csv('output/fastlmm-'+nameParm,header=0,index_col=None,sep='\t')
    
    df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
    df=df.sort_values(by='SNP')
    
    return({
        'traitInd':traitInd,
        'waldStat':(df['SNPWeight']/df['SNPWeightSE']).values.flatten(),
        'pLRT':chi2.sf(2*(df['AltLogLike']-df['NullLogLike']),1).flatten(),
        'pWald':df['Pvalue'].values.flatten(),
        'AltLogLike':df['AltLogLike'].values.flatten(),
        'beta':df['SNPWeight'].values.flatten(),
        'se':df['SNPWeightSE'].values.flatten()
    })
                                  
def runGemma(core,snp,trait,traitInd,parms):
    local=parms['local']
    
    nameParm=snp+'-'+trait+'-'+core
    cmd=[local+'ext/gemma','-bfile','ped/snp-'+snp,'-lmm','4','-o','gemma-'+nameParm,
         '-k','grm/gemma-'+snp,'-n',str(traitInd+1),'-c','ped/cov.txt','-p','ped/Y-'+trait+'.txt']
    
    subprocess.run(cmd) 

    df=pd.read_csv('output/gemma-'+nameParm+'.assoc.txt',header=0,index_col=None,sep='\t')
    
    return({
        'traitInd':traitInd,
        'waldStat':(df['beta']/df['se']).values.flatten(),
        'pLRT':df['p_lrt'].values.flatten(),
        'pWald':df['p_wald'].values.flatten(),
        'AltLogLike':df['logl_H1'].values.flatten(),
        'beta':df['beta'].values.flatten(),
        'se':df['se'].values.flatten()
    })
