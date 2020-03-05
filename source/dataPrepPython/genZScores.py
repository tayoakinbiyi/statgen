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
                #genZScoresHelp(str(core),str(snp),traitRange,parms,numSubjects)
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
    grm=parms['grm']
            
    waldStat=[]
    eta=[]
        
    if 'fast' in reg:
        assert ('ped' in reg) or ('bed' in reg)
        cmd=[local+'ext/fastlmmc','-covar','inputs/cov.phe','-maxThreads','1','-out','output/fastlmm-'+core,'-pheno',
             'inputs/Y.phe','-simLearnType','Full','-brentMinLogVal','-10','-brentMaxLogVal','10','-ML']
        if 'ped' in reg:
            cmd+=['-file','inputs/'+snp]
        if 'bed' in reg:
            cmd+=['-bfile','inputs/'+snp]
        if 'lmm' in reg:
            cmd+=['-eigen','grm/fast-eigen-'+snp]
        if 'lm' in reg:
            cmd+=['-linreg']        

    if 'gcta' in reg:
        assert ('bed' in reg)
        cmd=[local+'ext/gcta64','--qcovar','inputs/cov.phe','--out','output/gcta-'+core,'--pheno','inputs/Y.phe','--threads','1',
             '--bfile','inputs/'+snp,'--grm','grm/gcta-'+snp+'/grm','--reml-maxit','2000']

    if 'gemma' in reg:
        assert ('bimbam' in reg) or ('bed' in reg)
        cmd=[local+'ext/gemma','-o','gemma-'+core,'-c','inputs/cov.txt','-p','inputs/Y.phe']
        if 'bed' in reg:
            cmd+=['-bfile','inputs/'+snp]
        if 'bimbam' in reg:
            cmd+=['-g','inputs/'+snp+'.bimbam']
        if 'lmm' in reg:
            cmd+=['-lmm','4','-d','grm/gemma-eigen-'+snp+'/D','-u','grm/gemma-eigen-'+snp+'/U']
        if 'lm' in reg:
            cmd+=['-lm','4']
        
    if 'mixedlm' in reg:
        assert ('bimbam' in reg)
        cols=['L'+str(i) for i in range(numSubjects)]
        vc={'genotype':'~0+'+'+'.join(cols)}
        L=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
        y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2+traitRange]

        snpData=pd.DataFrame(np.loadtxt('inputs/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T,columns=['S'+str(i) for 
            i in range(numSnps)],dtype='int')

        data=pd.concat([pd.DataFrame(y,columns=['Y'+str(i) for i in traitRange]),pd.DataFrame(L,columns=cols),snpData],axis=1)
        data.insert(0,'groups',1)
    
    if 'limix' in reg:
        assert ('bimbam' in reg) or ('bed' in reg)
        if 'bed' in reg:
            _,_,bed=plink.read('inputs/'+snp)
            bimBamFmt=bed.compute().T
        if 'bimbam' in reg:
            bimBamFmt=np.loadtxt('inputs/'+snp+'.bimbam',delimiter='\t',dtype=str)[:,3:].T.astype(float)
        Y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2:]
        K=np.loadtxt('grm/limix-'+snp,delimiter='\t')
        M=np.ones([numSubjects,1])
        
    for traitInd in traitRange:
        print('core {} , {} of {}'.format(core,traitInd-min(traitRange),len(traitRange)),flush=True)
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

        if 'gcta' in reg:
            loopCmd=cmd+['--mlma','--mpheno',str(traitInd+1)]
            subprocess.call(loopCmd)

            df=pd.read_csv('output/gcta-'+core+'.mlma',header=0,index_col=None,sep='\t')
            tt=(df['b']/df['se']).values
            waldStat+=[norm.ppf(t.cdf(tt,numSubjects-2)).reshape(-1,1)]

            loopCmd=cmd+['--reml','--mpheno',str(traitInd+1)]
            subprocess.call(loopCmd)
            pdb.set_trace()
            etaDF=pd.read_csv('output/gcta-'+core+'.hsq',header=0,index_col=None,nrows=2,sep='\t')
            etaEst=etaDF.iloc[0,1]/(etaDF.iloc[1,0]+etaDF.iloc[1,1])

            eta+=[np.array([[etaEst]]*etaDF.shape[0])]
    
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
            
    waldStat=np.concatenate(waldStat,axis=1)
    eta=np.concatenate(eta,axis=1)
    
    return({'traitRange':traitRange,
            'waldStat':waldStat,
            'eta':eta
           })
