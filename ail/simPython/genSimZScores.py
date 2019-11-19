from ail.opPython.DB import *
import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
from ail.dataPrepPython.genZScores import *
import time
from statsmodels.regression.mixed_linear_model import MixedLM,MixedLMParams
from statsmodels.regression.linear_model import OLS
import warnings

def genSimZScores(parms):
    local=parms['local']
    name=parms['name']
    muEpsRange=parms['muEpsRange']
    H2SnpSize=parms['H2SnpSize']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    numCores=parms['numCores']

    DBCreateFolder('holds',parms)
        
    #genZScores(parms)
            
    H2SnpSet=pd.read_csv('ped/ail-2.ped',sep='\t',header=None)
    H2Map=pd.read_csv('ped/ail-2.map',sep='\t',header=None).iloc[:,1].values.flatten()
    
    traitData=pd.read_csv('ped/traitData',index_col=None,header=0,sep='\t')
    Y=np.loadtxt('ped/ail.phe',delimiter='\t')[:,2:]    

    N=Y.shape[1]
    tmpTrait=traitData.copy()
    tmpTrait.insert(0,'loc',range(tmpTrait.shape[0]))
    tmpTrait=tmpTrait.sort_values(by=['chr','loc'])
    tmpTraitSize=tmpTrait.groupby('chr')['loc'].count().values.tolist()
    tmpTrait.insert(0,'chrLoc',np.concatenate([range(x) for x in tmpTraitSize]))
    tmpTrait=tmpTrait.sort_values(by='loc')

    for k in range(len(muEpsRange)):
        mu=muEpsRange[k][0]
        eps=muEpsRange[k][1]
        nameParm=str(k)
        
        if os.path.exists('holds/genSimZScores-'+nameParm) or os.path.exists('score/'+str(3+k)+'-1'):
            continue

        np.savetxt('holds/genSimZScores-'+nameParm,np.array([]),delimiter='\t')

        eqtlList=pd.DataFrame(columns=['snp','loc','z'],index=range(H2SnpSize*eps))  
        count=0
        numLeft=len(H2Map)
        eqtlList=[]
        with ProcessPoolExecutor(numCores) as executor:
            snp=0
            while numLeft>0:
                futures=[]
                for core in range(min(numLeft,numCores)):
                    genSimZScoresHelp(parms,N,mu,eps,snp,H2SnpSet.iloc[:,6+snp],Y,H2Map[snp],nameParm)
                    futures+=[executor.submit(genSimZScoresHelp,parms,N,mu,eps,snp,H2SnpSet.iloc[:,6+snp],Y,H2Map[snp],nameParm)]
                    snp+=1
                    numLeft-=1

                for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                    eqtlList+=[f.result()]
        
        eqtlList=pd.concat(eqtlList,axis=0)
        
        for trait in traitChr:
            z=np.loadtxt('score/2-'+str(trait),delimiter='\t')
            tmpEqtlList=eqtlList[traitData['chr'].iloc[eqtlList['loc']].values==trait]
            
            xLoc=tmpEqtlList['snp'].values.flatten().astype(int)
            yLoc=tmpTrait['chrLoc'].iloc[tmpEqtlList['loc']].values.flatten().astype(int)
            z[xLoc,yLoc]=tmpEqtlList['z'].values.flatten()
            
            np.savetxt('score/'+str(3+k)+'-'+str(trait),z,delimiter='\t')
                 
        print('wrote '+str(mu)+' '+str(eps),flush=True)
                
    return()

def genSimZScoresHelp(parms,N,mu,eps,snp,snpVec,Y,snpID,nameParm):
    local=parms['local']
    
    eqtlList=pd.DataFrame(columns=['snp','loc','z'],index=range(eps))  
    print('mu:'+str(mu)+'-eps:'+str(eps)+'\tsnp:'+str(snp),flush=True)

    loc=np.random.choice(N,size=eps,replace=False)

    open('ped/extract-'+str(snp),'w+').write(str(snpID)+'\n')
    snpVec=snpVec.str.split(' ',expand=True).replace({'A':0,'G':1}).sum(axis=1).values.reshape(-1,1)         

    f=np.mean(snpVec)/2
    pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(Y))}),pd.DataFrame(Y[:,loc]+
        (mu/np.sqrt(2*eps*f*(1-f)))*np.matmul(snpVec,np.random.choice([-1,1],size=eps                   
        ).reshape(1,-1)))],axis=1).to_csv('ped/'+nameParm+'-'+str(snp)+'.phe',header=False,index=False,sep='\t')

    for ind in range(eps):
        cmd=[local+'ext/fastlmmc','-file','ped/ail-2','-pheno','ped/'+nameParm+'-'+str(snp)+'.phe','-eigen','ped/eigen-all',
             '-simLearnType','Once','-mpheno',str(ind+1),'-extract','ped/extract-'+str(snp),'-out',
             'fastlmm/'+nameParm+'-'+str(snp)+'-'+str(ind+1),'-maxThreads',str(1)]
        subprocess.call(cmd)

        df=pd.read_csv('fastlmm/'+nameParm+'-'+str(snp)+'-'+str(ind+1),header=0,index_col=None,sep='\t')
        eqtlList.iloc[ind,:]=[snp,loc[ind],(df['SNPWeight']/df['SNPWeightSE']).values[0]]
        
    DBLog(str(snp)+'\t'+str(mu)+'\t'+str(eps)+'\t'+str(eqtlList['z'].min())+'\t'+str(eqtlList['z'].max()),parms)
    
    return(eqtlList)