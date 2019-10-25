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
    numH2Segments=parms['numH2Segments']

    DBCreateFolder(name,'simZScores',parms)
        
    notDoneH0=True
    doScore=True
    while notDoneH0:
        if doScore:
            genZScores(parms)
            
        notDoneH0=False
        doScore=False

        for trait in traitChr:
            if notDoneH0:
                continue
                
            for snp in snpChr:
                print('genHZScores checking '+snp+'-'+trait,flush=True)
                if not DBIsFile(name+'score',snp+'-'+trait+'-'+nameParm,parms):
                    notDoneH0=True
                    continue     
                    
        if notDoneH0:
            time.sleep(180)
            
    print('loading data',flush=True)
    H2SnpSet=pd.read_csv('ped/ail-2.ped',sep='\t',header=None).iloc[:,3:].values
    traitData=DBRead('process/traitData',parms)
    Y=DBRead(name+'process/Y',parms)    
    print('loaded data',flush=True)

    N=Y.shape[1]

    segLen=int(np.ceil(H2SnpSize/numH2Segments))
    numSnps,M=H2SnpSet.shape

    tmpTrait=traitData.copy()
    tmpTrait.insert(0,'loc',range(tmpTrait.shape[0]))
    tmpTrait=tmpTrait.sort_values(by=['chr','loc'])
    tmpTraitSize=tmpTrait.groupby('chr')['loc'].count().values.tolist()
    tmpTrait.insert(0,'chrLoc',np.concatenate([range(x) for x in tmpTraitSize]))
    tmpTrait=tmpTrait.sort_values(by='loc')

    for k in range(len(muEpsRange)):
        mu=muEpsRange[k][0]
        eps=muEpsRange[k][1]
        nameParm=str(mu)+'-'+str(eps)
        
        if DBIsFile('holds/genSimZScores-'+nameParm,parms) or DBIsFile('finished/genSimZScores-'+nameParm,parms):
            continue

        DBWrite(np.array([]),name+'holds/genSimZScores-'+nameParm,parms,True)

        eqtlList=pd.DataFrame(columns=['snp','loc','z'])  
        count=0
        for snp in range(numSnps):
            print('mu:'+str(mu)+'-eps:'+str(eps)+'\tsnp:'+str(snp),flush=True)

            loc=np.random.choice(N,size=eps,replace=False)

            pd.DataFrame({'Family ID':0,'Individual ID':range(numSnps),'Paternal ID':0,'Maternal ID':0,
                'Sex':0,'Phenotype':0,'snp':H2SnpSet[snp]}).to_csv('ped/'+nameParm+'-'+snp,header=False,index=False)

            f=np.mean(H2SnpSet[snp])/2
            pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits))}),pd.DataFrame(Y[:,loc]+
                (mu/np.sqrt(2*eps*f*(1-f)))*np.matmul(H2SnpSet[snp].reshape(-1,1),np.random.choice([-1,1],size=eps                   
                ).reshape(1,-1)))],axis=1).to_csv('ped/'+nameParm+'-'+snp+'.phe',header=False,index=False)

            for ind in range(eps):
                cmd=[local+'ext/fastlmmc','-file','ped/2','-pheno','ped/'+nameParm+'-'+snp+'.phe','-eigen','ped/eigen-all',
                     '-simLearnType','Once','-mpheno',str(ind+1),'-out',
                     'fastlmm/'+nameParm+'-'+snp+'-'+str(ind+1),'-maxThreads',str(numCores)]
                subprocess.call(cmd)

                df=pd.read_csv('fastlmm/'+nameParm+'-'+snp+'-'+str(ind+1),header=0,index_col=None)
                eqtlList.iloc[count*eps+ind,:]=[snp,loc[ind],(df['SnpWeight']/df['SnpWeightSE']).values]
            count+=1

        for trait in traitChr:
            z=DBRead(name+'score/2-'+trait,parms,True)
            tmpEqtlList=eqtlList[traitData['chr'].iloc[eqtlList['loc']].values==trait]
            
            xLoc=tmpEqtlList['snp'].values.flatten().astype(int)
            yLoc=tmpTrait['chrLoc'].iloc[tmpEqtlList['loc']].values.flatten().astype(int)
            z[xLoc,yLoc]=tmpEqtlList['z'].values.flatten()
            
            DBWrite(z,name+'score/'+str(3+k)+'-'+trait,parms,True)
            
        DBWrite(np.array([]),name+'finished/genSimZScores-'+nameParm,parms,True)
        
        print('wrote '+nameParm+' trait',flush=True)
                
    return()
