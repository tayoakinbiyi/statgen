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
from pylmm.lmm import LMM, GWAS

def genZScores(parms,snpChr):
    numCores=parms['numCores']
    reg=parms['reg']
    etaSq=parms['parms'][-4]
    numSubjects=parms['parms'][-3]
    numTraits=parms['parms'][-2]
    numSnps=parms['parms'][-1]
    reg=parms['reg']
    
    if snpChr is None:
        snpChr=range(1,len(numSnps)+1)
   
    DBCreateFolder('output',parms)
        
    for snp in snpChr:
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
                #genZScoresHelp(str(core),str(snp),traitRange,parms,numSnp,numSubjects)
                futures+=[executor.submit(genZScoresHelp,str(core),str(snp),traitRange,parms,numSnp,numSubjects)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                ans=f.result()
                traitRange=ans['traitRange']                 
                waldStat[:,traitRange]=ans['waldStat']
                eta[:,traitRange]=ans['eta']
        
        np.savetxt('score/waldStat-'+str(snp),waldStat,delimiter='\t')
        np.savetxt('score/eta-'+str(snp),eta,delimiter='\t') 
                    
    return()

def genZScoresHelp(core,snp,traitRange,parms,numSnps,numSubjects):
    local=parms['local']
    reg=parms['reg']
            
    waldStat=[]
    eta=[]
        
    if 'fast' in reg:
        assert ('ped' in reg) or ('bed' in reg)
        lmm='fast'
        cmd=[local+'ext/fastlmmc','-covar','cov/cov.phe','-maxThreads','1','-out','output/fastlmm-'+core,'-pheno',
             'Y/Y.phe','-simLearnType','Full','-brentMinLogVal','-10','-brentMaxLogVal','10','-ML']
        if 'ped' in reg:
            cmd+=['-file','snps/'+snp]
        if 'bed' in reg:
            cmd+=['-bfile','snps/'+snp]
        if 'lmm' in reg:
            cmd+=['-eigen','grm/fast-eigen-'+snp]
        if 'lm' in reg:
            cmd+=['-linreg']        

    if 'gemma' in reg:
        assert ('bimbam' in reg) or ('bed' in reg)
        lmm='gemma'
        cmd=[local+'ext/gemma','-o','gemma-'+core,'-c','cov/cov.txt','-p','Y/Y.phe','-silence']
        if 'bed' in reg:
            cmd+=['-bfile','snps/'+snp]
        if 'bimbam' in reg:
            cmd+=['-g','snps/'+snp+'.bimbam']
        if 'lmm' in reg:
            cmd+=['-lmm','4','-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U']
        if 'lm' in reg:
            cmd+=['-lm','4']
        
    if 'mixedlm' in reg:
        assert ('bimbam' in reg)
        lmm='mixedlm'
        cols=['L'+str(i) for i in range(numSubjects)]
        vc={'genotype':'~0+'+'+'.join(cols)}
        L=np.loadtxt('grm/Lgrm-1',delimiter='\t')
        y=np.loadtxt('Y/Y.phe',delimiter='\t')[:,2+traitRange]

        snpData=pd.DataFrame(np.loadtxt('snps/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T,columns=['S'+str(i) for 
            i in range(numSnps)],dtype='int')

        data=pd.concat([pd.DataFrame(y,columns=['Y'+str(i) for i in traitRange]),pd.DataFrame(L,columns=cols),snpData],axis=1)
        data.insert(0,'groups',1)
    
    if 'limix' in reg:
        assert ('bimbam' in reg) or ('bed' in reg)
        lmm='limix'
        if 'bed' in reg:
            _,_,bed=plink.read('snps/'+snp)
            bimBamFmt=bed.compute().T
        if 'bimbam' in reg:
            bimBamFmt=np.loadtxt('snps/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float)
        Y=np.loadtxt('Y/Y.phe',delimiter='\t')[:,2:]
        K=np.loadtxt('grm/limix-'+snp,delimiter='\t')
        M=np.loadtxt('cov/cov.txt',delimiter='\t')
        
    if 'pylmm' in reg:
        assert ('bimbam' in reg) or ('bed' in reg)
        lmm='pylmm'
        if 'bed' in reg:
            _,_,bed=plink.read('snps/'+snp)
            bimBamFmt=bed.compute()
        if 'bimbam' in reg:
            bimBamFmt=np.loadtxt('snps/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].astype(float)
        Y=np.loadtxt('Y/Y.txt',delimiter='\t').reshape(numSubjects,-1)
        Kva=np.loadtxt('grm/pylmm-eigen-'+snp+'/Kva',delimiter='\t')
        Kve=np.loadtxt('grm/pylmm-eigen-'+snp+'/Kve',delimiter='\t')
        K=np.loadtxt('grm/pylmm-eigen-'+snp+'/K',delimiter='\t')
        M=np.loadtxt('cov/cov.txt',delimiter='\t').reshape(numSubjects,-1)
               
    for traitInd in traitRange:
        print('lmm: {} , core {} , {} of {}'.format(lmm,core,traitInd-min(traitRange),len(traitRange)),flush=True)
        if 'fast' in reg:
            loopCmd=cmd+['-mpheno',str(traitInd+1)]
            subprocess.call(loopCmd)

            df=pd.read_csv('output/fastlmm-'+core,header=0,index_col=None,sep='\t')
            df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
            df=df.sort_values(by='SNP')

            df.rename(columns={'SNPWeight':'SnpWeight','SNPWeightSE':'SnpWeightSE'},inplace=True)

            tt=(df['SnpWeight']/df['SnpWeightSE']).values
            waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]
            eta+=[(df['NullGeneticVar']/(df['NullGeneticVar']+df['NullResidualVar'])).values.reshape(-1,1)]

        if 'gemma' in reg:
            loopCmd=cmd+['-n',str(traitInd+3)]
            subprocess.run(loopCmd) 

            df=pd.read_csv('output/gemma-'+core+'.assoc.txt',header=0,index_col=None,sep='\t')

            tt=(df['beta']/df['se']).values
            waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]
            eta+=[(df['l_remle']/(1+df['l_remle'])).values.reshape(-1,1)]
            
        if 'mixedlm' in reg:
            waldStatAll=[]
            etaAll=[]
            for snpInd in range(numSnps):
                ret=MixedLM.from_formula('Y'+str(traitInd)+'~1+S'+str(traitInd), data, groups='groups',re_formula='~0',
                    vc_formula=vc).fit()
                waldStatAll+=[ret.fe_params[1]/ret.bse_fe[1]]
                etaAll+=[ret.vcomp[0]/(ret.vcomp[0]+ret.scale)]

            waldStat+=[waldStatAll]
            eta+=[etaAll]
            
        if 'limix' in reg:
            model=scan(bimBamFmt, Y[:,traitInd], 'normal', K, M=M,verbose=False)
            ret=model.effsizes['h2']
            waldStat+=[(ret.loc[ret['effect_type']=='candidate','effsize']/ret.loc[
                ret['effect_type']=='candidate','effsize_se']).values.reshape(-1,1)]
            eta+=[np.array([[model._h0._v0/(model._h0._v0+model._h0._v1)]]*numSnps)]
            
        if 'pylmm' in reg:
            hmax,beta,sigma,L1 = LMM(Y[:,traitInd],K,Kva=Kva,Kve=Kve,X0=M).fit()
            TS,PS = GWAS(Y[:,traitInd],bimBamFmt.T,K,Kva=Kva,Kve=Kve,X0=M,REML=False,refit=True)
            eta+=[np.array([[hmax]]*numSnps)]
            waldStat+=[np.array(TS).reshape(-1,1)]
           
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })
