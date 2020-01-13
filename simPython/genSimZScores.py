from opPython.DB import *
import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
import time
from statsmodels.regression.mixed_linear_model import MixedLM,MixedLMParams
from statsmodels.regression.linear_model import OLS
import warnings

from dataPrepPython.genZScores import *

def genSimZScores(parms):
    muEpsRange=parms['muEpsRange']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    numCores=parms['numCores']
    fastlmm=parms['fastlmm']
    numTraitChr=parms['numTraitChr']    
    numSnpChr=parms['numSnpChr']
        
    H0SnpChr=len(parms['SnpSize'])
    
    snpSet=pd.read_csv('ped/snp-'+str(H0SnpChr)+'.ped',sep='\t',header=None)
    snpSize=parms['SnpSize'][-1]
    
    traitData=pd.read_csv('ped/traitData',index_col=None,header=0,sep='\t')
    traitData=traitData[traitData['chr'].isin(traitChr)]
    snpData=pd.read_csv('ped/snpData',index_col=None,header=0,sep='\t')
    snpData=snpData[snpData['chr'].isin(snpChr)]
    
    maf=snpData['maf'].values.flatten().tolist()
    minor=snpData['minor'].values.flatten().tolist()
    mouseIds=np.loadtxt('ped/mouseIds',delimiter='\t').astype(int)
    
    Y=np.empty([len(mouseIds),len(traitData)])
    for trait in traitChr:
        Y[:,traitData['chr'].values.flatten()==trait]=np.loadtxt('ped/Y-'+str(trait)+'.txt',delimiter='\t')

    N=Y.shape[1]
    M=len(muEpsRange)

    for k in range(M):
        mu=muEpsRange[k][0]
        eps=muEpsRange[k][1]
        
        if os.path.exists('holds/genSimZScores-'+str(k)) or os.path.exists('score/'+str(numSnpChr+1+k)+'-1'):
            continue

        np.savetxt('holds/genSimZScores-'+str(k),np.array([]),delimiter='\t')

        eqtlList=[]
        with ProcessPoolExecutor(numCores) as executor:
            snp=0
            while snp<snpSize:
                futures=[]
                for core in range(min(snpSize-snp,numCores)):
                    #genSimZScoresHelp(parms,mu,eps,core,snp,numSnpChr+k*numCores+core+1,
                    #    numTraitChr+k*numCores+core+1,snpSet.iloc[:,6+snp],Y,maf[snp],minor[snp],mouseIds,fastlmm)
                    futures+=[executor.submit(genSimZScoresHelp,parms,mu,eps,core,snp,numSnpChr+k*numCores+core+1,
                        numTraitChr+k*numCores+core+1,snpSet.iloc[:,6+snp],Y,maf[snp],minor[snp],mouseIds,fastlmm)]
                    snp+=1

                for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                    eqtlList+=[f.result()]
           
        eqtlList=pd.concat(eqtlList,axis=0).sort_values(by=['snp','loc'])
        
        for trait in traitChr:
            z=np.loadtxt('score/waldStat-3-'+str(trait),delimiter='\t')
            tmpEqtlList=eqtlList[traitData['chr'].iloc[eqtlList['loc']].values==trait]
            
            xLoc=tmpEqtlList['snp'].values.flatten().astype(int)
            yLoc=traitData['traitSubset'].iloc[tmpEqtlList['loc']].values.flatten().astype(int)
            z[xLoc,yLoc]=tmpEqtlList['z'].values.flatten()
            
            np.savetxt('score/waldStat-'+str(3+1+k)+'-'+str(trait),z,delimiter='\t')
                 
        print('wrote '+str(mu)+' '+str(eps),flush=True)
                
    return()

def genSimZScoresHelp(parms,mu,eps,core,snpInd,snp,trait,snpVec,Y,maf,minor,mouseIds,fastlmm):
    local=parms['local']
    #pdb.set_trace()
    eqtlList=pd.DataFrame(columns=['snp','loc','z'],index=range(eps))  
    
    snpVecInt=snpVec.str.split(' ',expand=True).apply(lambda df,minor: (df!=minor).sum(),axis=1,args=(minor,)).values.reshape(-1,1)
    
    N=len(Y)
    loc=np.random.choice(N,size=eps,replace=False)
    fam=pd.DataFrame({'Family ID':mouseIds,'Individual ID':0,'Paternal ID':mouseIds,'Maternal ID':0,'Sex':1,'Phenotype':1})
        
    sign=np.random.choice([-1,1],size=eps).reshape(1,-1)
    Y=Y[:,loc]+(mu/np.sqrt(2*eps*maf*(1-maf)))*np.matmul(snpVecInt,sign)
    
    pd.concat([fam, pd.DataFrame({'snp':snpVec})],axis=1).to_csv('ped/snp-'+str(snp)+'.ped',sep='\t',header=False,index=False)
    pd.DataFrame({'chr':1,'ID':0,'genetic dist':1,'Mbp':1},index=[0]).to_csv(
        'ped/snp-'+str(snp)+'.map',sep='\t',header=False,index=False)
    subprocess.call([local+'ext/plink','--file','ped/snp-'+str(snp),'--out','ped/snp-'+str(snp),'--make-bed','--noweb'])
    
    np.savetxt('ped/Y-'+str(trait)+'.txt',Y,delimiter='\t')
    pd.concat([fam[['Family ID','Individual ID']],pd.DataFrame(Y)],axis=1).to_csv('ped/Y-'+str(trait)+'.phe',
        header=False,index=False,sep='\t')
    
    fileParm='mu:'+str(mu)+'-eps:'+str(eps)+'-core:'+str(core)+'-snp:'+str(snp)+'-trait:'+str(trait)

    for ind in range(eps):
        print(fileParm,flush=True)
        ans=genZScoresHelp(str(core),str(snp),str(trait),ind,parms,fastlmm)
        eqtlList.iloc[ind,:]=[snpInd,loc[ind],ans['waldStat'][0]]
    
    DBLog(str(snp)+'\t'+str(mu)+'\t'+str(eps)+'\t'+str(eqtlList['z'].min())+'\t'+str(eqtlList['z'].max()),parms)
    
    return(eqtlList)