import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from scipy.stats import chi2,t,norm

from opPython.DB import *
from opPython.verboseArrCheck import *

def genZScores(parms,snpChr):
    numCores=parms['numCores']
    reg=parms['reg']
    etaSq=parms['parms'][-4]
    numSubjects=parms['parms'][-3]
    numTraits=parms['parms'][-2]
    numSnps=parms['parms'][-1]
    reg=parms['reg']
   
    DBLog('genZScores')
    DBCreateFolder('output',parms)
        
    for snp in range(1,len(numSnps)+1):
        numSnp=numSnps[snp-1]
        pLRT=np.full([numSnp,numTraits],np.nan)
        pWald=np.full([numSnp,numTraits],np.nan)
        waldStat=np.full([numSnp,numTraits],np.nan)
        AltLogLike=np.full([numSnp,numTraits],np.nan)
        beta=np.full([numSnp,numTraits],np.nan)
        se=np.full([numSnp,numTraits],np.nan)
        eta=np.full([numSnp,numTraits],np.nan)
                        
        with ProcessPoolExecutor(numCores) as executor:
            futures=[]
            
            for core in range(numCores):
                traitRange=np.arange(core*int(np.ceil(numTraits/numCores)),min(numTraits,(core+1)*int(
                    np.ceil(numTraits/numCores))))
                if len(traitRange)==0:
                    continue
                #genZScoresHelp(str(core),str(snp),traitRange,parms,data,numSubjects)
                futures+=[executor.submit(genZScoresHelp,str(core),str(snp),traitRange,parms,numSubjects)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                ans=f.result()
                traitRange=ans['traitRange']
                   
                waldStat[:,traitRange]=ans['waldStat']
                pLRT[:,traitRange]=ans['pLRT']
                pWald[:,traitRange]=ans['pWald']
                AltLogLike[:,traitRange]=ans['AltLogLike']
                beta[:,traitRange]=ans['beta']
                se[:,traitRange]=ans['se']
                eta[:,traitRange]=ans['eta']
        
        np.savetxt('score/waldStat-'+str(snp),waldStat,delimiter='\t')
        np.savetxt('score/pLRT-'+str(snp),pLRT,delimiter='\t')
        np.savetxt('score/pWald-'+str(snp),pWald,delimiter='\t') 
        np.savetxt('score/AltLogLike-'+str(snp),AltLogLike,delimiter='\t') 
        np.savetxt('score/beta-'+str(snp),beta,delimiter='\t') 
        np.savetxt('score/se-'+str(snp),se,delimiter='\t') 
        np.savetxt('score/eta-'+str(snp),eta,delimiter='\t') 
                    
    return()

def genZScoresHelp(core,snp,traitRange,parms,numSubjects):
    reg=parms['reg']
    
    if 'fast' in reg:
        return(runFastlmm(core,snp,traitRange,parms,numSubjects))
    if 'gemma' in reg:
        return(runGemma(core,snp,traitRange,parms,numSubjects))        

def runFastlmm(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    eta=[]
    
    cmd=[local+'ext/fastlmmc','-covar','inputs/cov.phe','-maxThreads','1','-out','output/fastlmm-'+core,'-pheno','inputs/Y.phe']
    if 'ped' in reg:
        cmd+=['-file','inputs/'+snp]
    if 'bed' in reg:
        cmd+=['-bfile','inputs/'+snp]
    if 'lmm' in reg:
        cmd+=['-eigen','grm/fast-eigen-'+snp]
    if 'lm' in reg:
        cmd+=['-linreg']        

    for traitInd in traitRange:
        loopCmd=cmd+['-mpheno',str(traitInd+1)]
        print('fastlmm core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
        subprocess.call(loopCmd)
        
        df=pd.read_csv('output/fastlmm-'+core,header=0,index_col=None,sep='\t')
        df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
        df=df.sort_values(by='SNP')
        
        df.rename(columns={'SNPWeight':'SnpWeight','SNPWeightSE':'SnpWeightSE'},inplace=True)
    
        tt=(df['SnpWeight']/df['SnpWeightSE']).values
        waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]
        pLRT+=[chi2.sf(2*(df['AltLogLike']-df['NullLogLike']),1).reshape(-1,1)]
        pWald+=[df['Pvalue'].values.reshape(-1,1)]
        AltLogLike+=[df['AltLogLike'].values.reshape(-1,1)]
        beta+=[df['SnpWeight'].values.reshape(-1,1)]
        se+=[df['SnpWeightSE'].values.reshape(-1,1)]
        eta+=[(np.exp(df['NullLogDelta'])/(1+np.exp(df['NullLogDelta']))).values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    pLRT=np.concatenate(pLRT,axis=1)
    pWald=np.concatenate(pWald,axis=1)
    AltLogLike=np.concatenate(AltLogLike,axis=1)
    beta=np.concatenate(beta,axis=1)
    se=np.concatenate(se,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'pLRT':pLRT,
            'pWald':pWald,
            'AltLogLike':AltLogLike,
            'beta':beta,
            'se':se,
            'eta':eta
           })
                                  
def runGemma(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    eta=[]
    
    cmd=[local+'ext/gemma','-o','gemma-'+core,'-c','inputs/cov.txt','-p','inputs/Y.phe']
    if 'bed' in reg:
        cmd+=['-bfile','inputs/'+snp]
    if 'bimbam' in reg:
        cmd+=['-g','inputs/'+snp+'.bimbam']
    if 'lmm' in reg:
        cmd+=['-lmm','4','-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U']
    if 'lm' in reg:
        cmd+=['-lm','4']
    
    for traitInd in traitRange:
        loopCmd=cmd+['-n',str(traitInd+3)]
        
        print('gemma core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
        
        subprocess.run(loopCmd) 
        df=pd.read_csv('output/gemma-'+core+'.assoc.txt',header=0,index_col=None,sep='\t')
        tt=(df['beta']/df['se']).values
        waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]
        pLRT+=[df['p_lrt'].values.reshape(-1,1)]
        pWald+=[df['p_wald'].values.reshape(-1,1)]
        AltLogLike+=[df['p_wald'].values.reshape(-1,1)]#logl_H1
        beta+=[df['beta'].values.reshape(-1,1)]
        se+=[df['se'].values.reshape(-1,1)]
        eta+=[(df['l_remle']/(1+df['l_remle'])).values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    pLRT=np.concatenate(pLRT,axis=1)
    pWald=np.concatenate(pWald,axis=1)
    AltLogLike=np.concatenate(AltLogLike,axis=1)
    beta=np.concatenate(beta,axis=1)
    se=np.concatenate(se,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'pLRT':pLRT,
            'pWald':pWald,
            'AltLogLike':AltLogLike,
            'beta':beta,
            'se':se,
            'eta':eta
           })