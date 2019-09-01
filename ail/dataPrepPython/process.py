import pandas as pd
import numpy as np
import pdb
from collections import Counter
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pyreadr
import subprocess
from sklearn.metrics import r2_score

from ail.opPython.DB import *

def process(parms):
    response=parms['response']
    name=parms['name']
    local=parms['local']
    
    DBSyncLocal('data',parms)

    numPCs=parms['numPCs']
    traitChr=parms['traitChr']
    
    remPCFromTraits=parms['remPCFromTraits']
    remPCFromSnp=parms['remPCFromSnp']
    remPCCorrSnp=parms['remPCCorrSnp']
    PCIsPreds=parms['PCIsPreds']
    CovIsPreds=parms['CovIsPreds']
    remCovFromTraits=parms['remCovFromTraits']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten()

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']

    covariates=pd.read_csv(local+'data/ail.phenos.final.txt',sep='\t',header=0,index_col=0)
    allIds=covariates.index.values.flatten().astype(int)

    snps=pd.read_csv(local+'data/'+parms['snpFile'],sep='\t',header=None,index_col=None)
    snps.columns=np.append(['chr','Mbp','minor','major'],allIds)
    snpData=snps[['chr','Mbp']]
    snps.loc[:,'Mbp']=range(len(snps))
    snps=snps[np.append(['Mbp','minor','major'],mouseIds)]
    
    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':'chr'+mouseGenes['chrom'].astype(str),
        'Mbp':(mouseGenes['cds_start']+mouseGenes['cds_end'])/2})
    
    traitData=traitData[traitData['chr'].isin(traitChr)]
    
    traitData=(pd.DataFrame({'trait':traits.columns}).merge(traitData,on='trait'))[['trait','chr','Mbp']]
    
    traits=traits[traitData['trait']].values
    
    if quantNormalizeExpr:
        traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))   

    covariates=covariates.loc[mouseIds,['sex','batch']]
    covariates=pd.concat([covariates,pd.get_dummies(covariates['batch'],drop_first=True)],axis=1).drop(columns='batch')
    covariates.insert(0,'intercept',1)
    
    if remCovFromTraits:
        reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(covariates,traits)
        traits=(traits-reg.predict(covariates))    
    
    np.savetxt(local+name+'process/dummy.txt',np.ones([len(mouseIds),1]),delimiter='\t')
    DBUpload(name+'process/dummy.txt',parms,toPickle=False)
        
    print('start grm creation',flush=True)
    for snp in parms['snpChr']:
        genGRMHelp(snp,snps[snpData['chr']!=snp],parms)

    print('grm finished '+snp,flush=True)
    U,D,Vt=np.linalg.svd(traits)
    PCs=np.concatenate([np.ones([len(traits),1]),U[:,0:numPCs]],axis=1)

    if remPCFromTraits:
        reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(PCs,traits)
        traits=(traits-reg.predict(PCs))    

    header=snps.T.iloc[0:3,:]
    snpY=snps.T.values[3:,:]

    reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=parms['cpu']).fit(PCs,snpY)
    snpYHat=reg.predict(PCs)
    R2=r2_score(snpY,snpYHat,multioutput='raw_values').flatten()
    print('r2 calculated',flush=True)

    toKeep=(R2<.9)
    print(len(toKeep)-sum(toKeep),' snps removed')

    if remPCFromSnp:
        snps=pd.concat([header,pd.DataFrame(snpY-snpYHat)],axis=0).T

    if remPCCorrSnp:
        snps=snps.loc[toKeep,:]
        R2=R2[toKeep]
        snpData=snpData[toKeep]
            
    DBWrite(R2.flatten(),name+'process/snpR2',parms)
        
    if PCIsPreds:
        np.savetxt(local+name+'process/preds.txt',PCs,delimiter='\t')
    elif CovIsPreds:
        np.savetxt(local+name+'process/preds.txt',covariates,delimiter='\t')
    else:
        np.savetxt(local+name+'process/preds.txt',np.ones([len(mouseIds),1]),delimiter='\t')       
    DBUpload(name+'process/preds.txt',parms,toPickle=False)

    DBWrite(traitData,name+'process/traitData',parms)

    for trait in parms['traitChr']:
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',traits[:,traitData['chr']==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
                    
    DBWrite(snpData,name+'process/snpData',parms)
    
    for snp in parms['snpChr']:
        snps[snpData['chr']==snp].to_csv(local+name+'process/geno-'+snp+'.txt',sep='\t',index=False,header=False)        
        DBUpload(name+'process/geno-'+snp+'.txt',parms,toPickle=False)
        
    return()

def genGRMHelp(snp,snps,parms):
    local=parms['local']
    name=parms['name']
    grmParm=parms['grmParm']
    
    if DBIsFile(name+'process','grm-'+snp+'.txt',parms):
        return()
    
    snps.to_csv('geno-grm-'+snp+'.txt',index=None,header=None,sep='\t')
    
    # generate loco
    print('grm',snp,grmParm,flush=True)
    subprocess.run(['./gemma','-g','geno-grm-'+snp+'.txt','-gk',('1' if grmParm=='c' else '2'),
        '-o','grm-'+snp,'-p',local+name+'process/dummy.txt'])

    # move grm to scratch
    os.rename('output/grm-'+snp+'.'+grmParm+'XX.txt',local+name+'process/grm-'+snp+'.txt')

    os.remove('geno-grm-'+snp+'.txt')
    DBUpload(name+'process/grm-'+snp+'.txt',parms,toPickle=False)
    #os.remove('output/grm-'+snp+'.log.txt')    