from ail.opPython.DB import *
import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
from ail.dataPrepPython.score import *
import time
from statsmodels.regression.mixed_linear_model import MixedLM

def genSimZScoresTest(parms):
    local=parms['local']
    name=parms['name']
    muEpsRange=parms['muEpsRange']
    H1SnpSize=parms['H1SnpSize']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    numCores=parms['numCores']

    DBSyncLocal(name+'process',parms)
    
    H1SnpSet=pd.read_csv(local+name+'process/geno-chr1.txt',sep='\t',header=None).iloc[:,3:].values
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    Y=DBRead(name+'process/Y',parms,toPickle=True)
    vc={'1':{1:DBRead(name+'process/LgrmAll',parms,toPickle=True)}}
    N=Y.shape[1]
    
    nameParm='mu:0-eps:0'
    
    notDoneH0=True
    doScore=True
    while notDoneH0:
        if doScore:
            score(parms,nameParm)
            
        notDoneH0=False
        doScore=False

        for trait in traitChr:
            if notDoneH0:
                continue
                
            for snp in snpChr:
                print('genHZScores checking '+snp+'-'+trait,flush=True)
                if not DBIsFile(name+'score','p-'+snp+'-'+trait+'-'+nameParm,parms):
                    notDoneH0=True
                    continue     
                    
        if notDoneH0:
            time.sleep(180)
            
    for muEps in muEpsRange:
        mu=muEps[0]
        eps=muEps[1]
        
        nameParm='mu:'+str(mu)+'-eps:'+str(eps)
        
        if DBIsFile(name+'holds','genHZScores-'+nameParm,parms):
            continue
            
        DBWrite(np.array([]),name+'holds/genHZScores-'+nameParm,parms,True)
        print('genHZScores '+nameParm,flush=True)            

        futures=[]
        eqtlList=[]
        
        #genH1ZScoreHelp(mu,eps,parms,0,H1SnpSet[0,:],Y,traitData)
        snpsPerCore=int(np.ceil(H1SnpSize/numCores))
        with ProcessPoolExecutor(numCores) as executor: 
            for core in range(numCores):
                snpRange=range(core*snpsPerCore,min((core+1)*snpsPerCore,H1SnpSize))
                genHZScoreHelp(mu,eps,parms,snpRange,H1SnpSet[snpRange,:],Y,N,vc)
                futures.append(executor.submit(genHZScoreHelp,mu,eps,snpRange,H1SnpSet[snpRange,:],Y,N,vc))

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                eqtlList+=f.result()

        eqtlList=pd.DataFrame(eqtlList,columns=['snp','loc','z'])            

        tmpTrait=traitData.copy()
        tmpTrait.insert(0,'loc',range(tmpTrait.shape[0]))
        tmpTrait=tmpTrait.sort_values(by=['chr','loc'])
        tmpTraitSize=tmpTrait.groupby('chr')['loc'].count().values.tolist()
        tmpTrait.insert(0,'chrLoc',np.concatenate([range(x) for x in tmpTraitSize]))
        tmpTrait=tmpTrait.sort_values(by='loc')

        for trait in traitChr:
            z=DBRead(name+'score/p-chr1-'+trait+'-mu:0-eps:0',parms,True)
            tmpEqtlList=eqtlList[traitData['chr'].iloc[eqtlList['loc']].values==trait]
            
            xLoc=tmpEqtlList['snp'].values.flatten()
            yLoc=tmpTrait['chrLoc'].iloc[tmpEqtlList['loc']].values.flatten()
            z[xLoc,yLoc]=tmpEqtlList['z'].values.flatten()
            
            DBWrite(z,name+'score/p-chr1-'+trait+'-'+nameParm,parms,True)
                
    return()

def genHZScoreHelp(mu,eps,snpRange,H1SnpSet,Y,N,vc):
    eqtlList=[]
    pdb.set_trace()
    for snp in range(len(snpRange)):
        if snp%50==0:
            print('mu:'+str(mu)+'-eps:'+str(eps)+'\t'+str(snp),flush=True)
        loc=np.random.choice(N,size=eps,replace=False)
        
        snpVec=H1SnpSet[snp,:].reshape(-1,1)
        
        f=np.mean(snpVec)/2
        pheno=Y[:,eqtlList['loc']].copy()+(mu/np.sqrt(2*eps*f*(1-f)))*np.matmul(snpVec,np.random.choice([-1,1],size=eps).reshape(1,-1))

        pred=np.concatenate([np.ones([len(Y),1]),snpVec],axis=1)

        val=[]
        for ind in range(eps):
            res=MixedLM(pheno[:,ind], pred, [1]*len(Y),exog_vc=vc).fit()
            eqtlList+=[[snpRange[snp],loc,res.fe_params[1]/res.bse_fe[1]]]
            
    return(eqtlList)
