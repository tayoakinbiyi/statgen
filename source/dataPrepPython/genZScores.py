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
    data=parms['data']
    sim=parms['sim']
    etaSq=sim[-4]
    numSubjects=sim[-3]
    numTraits=sim[-2]
    numSnps=sim[-1]
    
    DBLog('genZScores')
    DBCreateFolder('output',parms)
        
    for snp in snpChr:
        numSnp=numSnps[snp-1]
        pLRT=np.full([numSnp,numTraits],np.nan)
        pWald=np.full([numSnp,numTraits],np.nan)
        waldStat=np.full([numSnp,numTraits],np.nan)
        AltLogLike=np.full([numSnp,numTraits],np.nan)
        beta=np.full([numSnp,numTraits],np.nan)
        se=np.full([numSnp,numTraits],np.nan)
                        
        with ProcessPoolExecutor(numCores) as executor:
            futures=[]
            
            for core in range(numCores):
                traitRange=np.arange(core*int(np.ceil(numTraits/numCores)),min(numTraits,(core+1)*int(
                    np.ceil(numTraits/numCores))))
                if len(traitRange)==0:
                    continue
                #genZScoresHelp(str(core),str(snp),traitRange,parms,data,numSubjects)
                futures+=[executor.submit(genZScoresHelp,str(core),str(snp),traitRange,parms,data,numSubjects)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                ans=f.result()
                traitRange=ans['traitRange']
                   
                waldStat[:,traitRange]=ans['waldStat']
                pLRT[:,traitRange]=ans['pLRT']
                pWald[:,traitRange]=ans['pWald']
                AltLogLike[:,traitRange]=ans['AltLogLike']
                beta[:,traitRange]=ans['beta']
                se[:,traitRange]=ans['se']
        
        op='fast' if 'fast' in data else 'gemma'
        np.savetxt('score/'+op+'-waldStat-'+str(snp),waldStat,delimiter='\t')
        np.savetxt('score/'+op+'-pLRT-'+str(snp),pLRT,delimiter='\t')
        np.savetxt('score/'+op+'-pWald-'+str(snp),pWald,delimiter='\t') 
        np.savetxt('score/'+op+'-AltLogLike-'+str(snp),AltLogLike,delimiter='\t') 
        np.savetxt('score/'+op+'-beta-'+str(snp),beta,delimiter='\t') 
        np.savetxt('score/'+op+'-se-'+str(snp),se,delimiter='\t') 
            
        os.symlink(op+'-waldStat-'+str(snp),'score/waldStat-'+str(snp))
        
    return()

def genZScoresHelp(core,snp,traitRange,parms,data,numSubjects):      
    if 'fast' in data:
        return(runFastlmm(core,snp,traitRange,parms,numSubjects,data))
    if 'gemma' in data:
        return(runGemma(core,snp,traitRange,parms,numSubjects,data))        

def runFastlmm(core,snp,traitRange,parms,numSubjects,data):
    simLearnType=parms['simLearnType']
    local=parms['local']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    
    cmd=[local+'ext/fastlmmc','-covar','inputs/cov.phe','-maxThreads','1','-simLearnType',simLearnType,
         '-out','output/fastlmm-'+core,'-pheno','inputs/Y.phe']
    if 'ped' in data:
        cmd+=['-file','inputs/'+snp]
    if 'bed' in data:
        cmd+=['-bfile','inputs/'+snp]
    if 'lmm' in data:
        cmd+=['-eigen','grm/fast-eigen-'+snp]
    if 'lm' in data:
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
                                  
def runGemma(core,snp,traitRange,parms,numSubjects,data):
    local=parms['local']
        
    waldStat=[]
    pLRT=[]
    pWald=[]
    AltLogLike=[]
    beta=[]
    se=[]
    
    cmd=[local+'ext/gemma','-o','gemma-'+core,'-c','inputs/cov.txt','-p','inputs/Y.phe']
    if 'ped' in data:
        cmd+=['-bfile','inputs/'+snp]
    if 'bimbam' in data:
        cmd+=['-g','inputs/'+snp+'.bimbam']
    if 'lmm' in data:
        cmd+=['-lmm','4','-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U']
    if 'lm' in data:
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