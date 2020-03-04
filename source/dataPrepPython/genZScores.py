import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from scipy.stats import chi2,t,norm
from limix.qtl import scan
from opPython.DB import *
from opPython.verboseArrCheck import *
from statsmodels.regression.mixed_linear_model import *
from limix.io import plink

def genZScores(parms,snpChr):
    numCores=parms['numCores']
    reg=parms['reg']
    etaSq=parms['parms'][-4]
    numSubjects=parms['parms'][-3]
    numTraits=parms['parms'][-2]
    numSnps=parms['parms'][-1]
    reg=parms['reg']
   
    DBCreateFolder('output',parms)
        
    for snp in range(1,len(numSnps)+1):
        numSnp=numSnps[snp-1]
        waldStat=np.full([numSnp,numTraits],np.nan)
        eta=np.full([numSnp,numTraits],np.nan)
                        
        with ProcessPoolExecutor(numCores) as executor:
            futures=[]
            
            for core in range(numCores):
                traitRange=np.arange(core*int(np.ceil(numTraits/numCores)),min(numTraits,(core+1)*int(
                    np.ceil(numTraits/numCores))))
                if len(traitRange)==0:
                    continue
                #genZScoresHelp(str(core),str(snp),traitRange,parms,numSubjects)
                futures+=[executor.submit(genZScoresHelp,str(core),str(snp),traitRange,parms,numSubjects)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                ans=f.result()
                traitRange=ans['traitRange']                 
                waldStat[:,traitRange]=ans['waldStat']
                eta[:,traitRange]=ans['eta']
        
        np.savetxt('score/waldStat-'+str(snp),waldStat,delimiter='\t')
        np.savetxt('score/eta-'+str(snp),eta,delimiter='\t') 
                    
    return()

def genZScoresHelp(core,snp,traitRange,parms,numSubjects):
    reg=parms['reg']
    grm=parms['grm']
    
    assert ('fast' in reg) or ('gemma' in reg) or ('mixedLM' in reg) or ('limix' in reg) or ('gcta' in reg)
    
    if 'fast' in reg:
        return(runFastlmm(core,snp,traitRange,parms,numSubjects))
    if 'gemma' in reg:
        return(runGemma(core,snp,traitRange,parms,numSubjects))  
    if 'mixedlm' in reg:
        return(runMixedLM(core,snp,traitRange,parms,numSubjects))
    if 'limix' in reg:
        return(runLimix(core,snp,traitRange,parms,numSubjects))
    if 'gcta' in reg:
        return(runGCTA(core,snp,traitRange,parms,numSubjects))

def runFastlmm(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
    
    assert ('bed' in reg) or ('ped' in reg)
        
    waldStat=[]
    eta=[]
    
    cmd=[local+'ext/fastlmmc','-covar','inputs/cov.phe','-maxThreads','1','-out','output/fastlmm-'+core,'-pheno',
         'inputs/Y.phe','-simLearnType','Full','-brentMinLogVal','-10','-brentMaxLogVal','10','-REML']
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
        eta+=[(df['NullGeneticVar']/(df['NullGeneticVar']+df['NullResidualVar'])).values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })
                                  
def runGCTA(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
    grm=parms['grm']
        
    assert ('bed' in reg) and ('gcta' in grm)
    waldStat=[]
    eta=[]

    cmd=[local+'ext/gcta64','--qcovar','inputs/cov.phe','--out','output/gcta-'+core,'--pheno','inputs/Y.phe','--threads','1',
         '--bfile','inputs/'+snp,'--grm','grm/gcta-'+snp+'/grm','--reml-maxit','2000']

    for traitInd in traitRange:
        loopCmd=cmd+['--mlma','--mpheno',str(traitInd+1)]
        print('gcta core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
        subprocess.call(loopCmd)
        
        df=pd.read_csv('output/gcta-'+core+'.mlma',header=0,index_col=None,sep='\t')
        tt=(df['b']/df['se']).values
        waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]
            
    waldStat=np.concatenate(waldStat,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':np.ones([len(waldStat),len(traitRange)])
           })

def runGemma(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
        
    waldStat=[]
    eta=[]
    assert ('bed' in reg) or ('bimbam' in reg)
    
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
        eta+=[(df['l_remle']/(1+df['l_remle'])).values.reshape(-1,1)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })

def runMixedLM(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
    numSnps=parms['parms'][-1][int(snp)-1]
        
    assert 'bimbam' in reg

    waldStat=[]
    eta=[]
    
    cols=['L'+str(i) for i in range(numSubjects)]
    vc={'genotype':'~0+'+'+'.join(cols)}
    L=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
    y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2+traitRange]
        
    snpData=pd.DataFrame(np.loadtxt('inputs/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T,columns=['S'+str(i) for 
        i in range(numSnps)],dtype='int')
        
    data=pd.concat([pd.DataFrame(y,columns=['Y'+str(i) for i in traitRange]),pd.DataFrame(L,columns=cols),snpData],axis=1)
    data.insert(0,'groups',1)

    for traitInd in traitRange:
        waldStatAll=[]
        etaAll=[]
        print('core {} traitInd {}'.format(core,traitInd))
        
        for snpInd in range(numSnps):
            ret=MixedLM.from_formula('Y'+str(traitInd)+'~1+S'+str(traitInd), data, groups='groups',re_formula='~0',
                vc_formula=vc).fit()
            waldStatAll+=[ret.fe_params[1]/ret.bse_fe[1]]
            etaAll+=[ret.vcomp[0]/(ret.vcomp[0]+ret.scale)]

        waldStat+=[waldStatAll]
        eta+=[etaAll]
    
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })

def runLimix(core,snp,traitRange,parms,numSubjects):
    local=parms['local']
    reg=parms['reg']
    numSnps=parms['parms'][-1][int(snp)-1]
    grm=parms['grm']
    
    assert ('bed' in reg) or ('bimbam' in reg)
        
    waldStat=[]
    eta=[]
        
    if 'bed' in reg:
        _,_,bed=plink.read('inputs/'+snp)
        bimBamFmt=bed.compute().T
    if 'bimbam' in reg:
        bimBamFmt=np.loadtxt('inputs/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float)
        
    Y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2:]
    K=np.loadtxt('grm/limix-'+snp,delimiter='\t')
    M=np.ones([numSubjects,1])

    for traitInd in traitRange:
        print('core {} traitInd {}'.format(core,traitInd))
        
        model=scan(bimBamFmt, Y[:,traitInd], 'normal', K, M=M,verbose=False)
        ret=model.effsizes['h2']
        waldStat+=[(ret.loc[ret['effect_type']=='candidate','effsize']/ret.loc[
            ret['effect_type']=='candidate','effsize_se']).values.reshape(-1,1)]
        eta+=[np.array([[model._h0._v0/(model._h0._v0+model._h0._v1)]]*numSnps)]
    
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })
