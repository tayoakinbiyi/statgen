from ail.opPython.DB import *
import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import subprocess

def genH1ZScores(parms):
    local=parms['local']
    name=parms['name']
    muList=parms['muList']
    epsList=parms['epsList']
    H1SnpSize=parms['H1SnpSize']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']

    DBSyncLocal(name+'process',parms)
    
    H1SnpSet=pd.read_csv(local+name+'process/geno-chr1.txt',sep='\t',header=None).iloc[:,3:].values
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    Y=DBRead(name+'process/Y',parms,toPickle=True)
    
    for mu in muList:
        for epsilon in epsList:
            futures=[]
            eqtlList=[]
                
            #genH1ZScoreHelp(mu,epsilon,parms,0,H1SnpSet[0,:],Y,traitData)
            with ProcessPoolExecutor(parms['cpu']) as executor: 
                for snp in range(H1SnpSize):
                    snpVec=H1SnpSet[snp,:]
                    futures.append(executor.submit(genH1ZScoreHelp,mu,epsilon,parms,snp,snpVec,Y,traitData))

                for f in as_completed(futures):
                    eqtlList+=[f.result()]
            
            eqtlList=pd.concat(eqtlList,axis=0)            
            
            for trait in traitChr:
                z=DBRead(name+'score/p-chr0-'+trait,parms,True)
                tmpEqtlList=eqtlList[eqtlList['trait']==trait]
                z[tmpEqtlList['snp'],tmpEqtlList['chrLoc']]=tmpEqtlList['z']
                DBWrite(z,name+'score/p-chr0-'+trait+'-'+str(mu)+'-'+str(epsilon),parms,True)
                
    return()

def genH1ZScoreHelp(mu,epsilon,parms,snp,snpVec,Y,traitData): 
    local=parms['local']
    name=parms['name']
    wald=parms['wald']

    pval='1' if wald else '2'    
    
    f=np.mean(snpVec)/2

    loc=np.random.choice(traitData.shape[0],size=epsilon,replace=False)
    pos=np.random.choice(loc,size=int(epsilon/2),replace=False).tolist()
    neg=list(set(loc)-set(pos))
    
    eqtlList=pd.DataFrame([[1,x] for x in pos] +[[-1,x] for x in neg],columns=['mul','loc'])
    eqtlList.insert(0,'trait',traitData['chr'].iloc[eqtlList['loc']].values)
    eqtlList.insert(0,'snp',snp)
    
    tmpTrait=traitData.copy()
    tmpTrait.insert(0,'loc',range(tmpTrait.shape[0]))
    tmpTrait=tmpTrait.sort_values(by=['chr','loc'])
    tmpTraitSize=tmpTrait.groupby('chr')['loc'].count().values.tolist()
    tmpTrait.insert(0,'chrLoc',np.concatenate([range(x) for x in tmpTraitSize]))
    tmpTrait=tmpTrait.sort_values(by='loc')
    
    eqtlList.insert(0,'chrLoc',tmpTrait['chrLoc'].iloc[eqtlList['loc'].values].values) # snp, mul, loc, trait, chrLoc
       
    snp=str(snp)
    mu=str(mu)
    epsilon=str(epsilon)
    
    path=local+name
    
    snpDF=pd.DataFrame([[0,'G','T']+snpVec.tolist()],index=[0]).to_csv(
        path+'H1/snp-'+snp+'-'+mu+'-'+epsilon+'.txt',sep='\t',index=False,header=False)
    
    val=[]
    for _,eqtl in eqtlList.iterrows():
        pheno=Y[:,eqtl['loc']]+(float(mu)/np.sqrt(2*int(epsilon)*f*(1-f)))*eqtl['mul']*snpVec
        np.savetxt(path+'H1/Y-'+snp+'-'+mu+'-'+epsilon+'.txt',pheno,delimiter='\t')

        cmd=[local+'ext/gemma','-g',path+'H1/snp-'+snp+'-'+mu+'-'+epsilon+'.txt','-p',
             path+'H1/Y-'+snp+'-'+mu+'-'+epsilon+'.txt','-lmm',pval,'-o',
             name[:-1]+'-'+snp+'-'+mu+'-'+epsilon,'-k',path+'process/grm-all.txt','-n','1',
             '-c',path+'process/dummy.txt','-silence']

        subprocess.run(cmd) 

        df=pd.read_csv('output/'+name[:-1]+'-'+snp+'-'+mu+'-'+epsilon+'.assoc.txt',sep='\t')
        os.remove('output/'+name[:-1]+'-'+snp+'-'+mu+'-'+epsilon+'.assoc.txt')

        val+=[(df['beta']/df['se']).iloc[0]]
    
    eqtlList.insert(0,'z',val)
    eqtlList=eqtlList.drop(columns='loc')                                          
    
    return(eqtlList)
