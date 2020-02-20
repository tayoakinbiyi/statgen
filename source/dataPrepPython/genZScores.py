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
    lmm=parms['lmm']
    N=parms['numSubjects']
    
    DBLog('genZScores')
    DBCreateFolder('output',parms)
    
    traitData=pd.read_csv('ped/traitData',index_col=None,header=0,sep='\t')
    traitData=traitData[traitData['chr'].isin(traitChr)]
    
    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpData=snpData[snpData['chr'].isin(snpChr)]
        
    numTraits=traitData.shape[0]

    for snp in snpChr:
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
                genZScoresHelp(str(core),str(snp),traitRange,parms,lmm,N)
                futures+=[executor.submit(genZScoresHelp,str(core),str(snp),traitRange,parms,lmm,N)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                ans=f.result()
                traitRange=ans['traitRange']
                    
                waldStat[:,traitRange]=ans['waldStat']
                pLRT[:,traitRange]=ans['pLRT']
                pWald[:,traitRange]=ans['pWald']
                AltLogLike[:,traitRange]=ans['AltLogLike']
                beta[:,traitRange]=ans['beta']
                se[:,traitRange]=ans['se']
        
        op='fast' if 'fast' in lmm else 'gemma'
        np.savetxt('score/'+op+'-waldStat-'+str(snp),waldStat,delimiter='\t')
        np.savetxt('score/'+op+'-pLRT-'+str(snp),pLRT,delimiter='\t')
        np.savetxt('score/'+op+'-pWald-'+str(snp),pWald,delimiter='\t') 
        np.savetxt('score/'+op+'-AltLogLike-'+str(snp),AltLogLike,delimiter='\t') 
        np.savetxt('score/'+op+'-beta-'+str(snp),beta,delimiter='\t') 
        np.savetxt('score/'+op+'-se-'+str(snp),se,delimiter='\t') 
            
        os.symlink(op+'-waldStat-'+str(snp),'score/waldStat-'+str(snp))
        
    return()

def genZScoresHelp(core,snp,traitRange,parms,lmm,N):      
    if 'fast' in lmm:
        return(runFastlmm(core,snp,traitRange,parms,N,lmm))
    else:
        return(runGemma(core,snp,traitRange,parms,N,lmm))        

def runFastlmm(core,snp,traitRange,parms,N,lmm):
    simLearnType=parms['simLearnType']
    local=parms['local']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    pdb.set_trace()
    
    cmd=[local+'ext/fastlmmc','-covar','inputs/cov.phe','-maxThreads','1','-simLearnType',simLearnType,
         '-out','output/fastlmm-'+core,'-pheno','ped/'+snp+'.phe']
    if 'ped' in lmm:
        cmd+=['-file','inputs/'+snp]
    if 'bed' in lmm:
        cmd+=['-bfile','inputs/'+snp]
    if 'lmm' in lmm:
        cmd+=['-eigen','grm/fast-eigen-'+snp]
    if 'lm' in lmm:
        cmd+=['-linreg']        

    for traitInd in traitRange:
        loopCmd=cmd+['-mpheno',str(traitInd+1)]
        print('fastlmm core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
        subprocess.call(loopCmd)
        
        df=pd.read_csv('output/fastlmm-'+core,header=0,index_col=None,sep='\t')
        df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
        df=df.sort_values(by='SNP')
    
        tt=(df['SNPWeight']/df['SNPWeightSE']).values
        waldStat+=[norm.ppf(t.cdf(tt,N-2)).reshape(-1,1)]
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
                                  
def runGemma(core,snp,traitRange,parms,N,lmm):
    local=parms['local']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    
    cmd=[local+'ext/gemma','-o','gemma-'+core,'-c','inputs/cov.txt','-p','inputs/Y.phe']
    if 'bed' in lmm:
        cmd+=['-bfile','inputs/'+snp]
    if 'bimbam' in lmm:
        cmd+=['-g','inputs/'+snp+'.bim']
    if 'lmm' in lmm:
        cmd+=['-lmm','4','-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U']
    if 'lm' in lmm:
        cmd+=['-lm','4']
    
    for traitInd in traitRange:
        loopCmd=cmd+['-n',str(traitInd+3)]
        
        print('gemma core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
        subprocess.run(loopCmd) 
        df=pd.read_csv('output/gemma-'+core+'.assoc.txt',header=0,index_col=None,sep='\t')
        tt=(df['beta']/df['se']).values
        waldStat+=[norm.ppf(t.cdf(tt,N-2)).reshape(-1,1)]
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