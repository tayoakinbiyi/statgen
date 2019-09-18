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
from ail.genPython.makePSD import *

def process(parms):
    response=parms['response']
    name=parms['name']
    local=parms['local']
    allChrGRM=parms['allChrGRM']
    
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    
    DBSyncLocal('data',parms)

    numPCs=parms['numPCs']
    
    remPCFromTraits=parms['remPCFromTraits']
    remCovFromTraits=parms['remCovFromTraits']

    remPCFromSnp=parms['remPCFromSnp']
    remPCCorrSnp=parms['remPCCorrSnp']

    PCIsPreds=parms['PCIsPreds']
    CovIsPreds=parms['CovIsPreds']
    quantNormalizeExpr=parms['quantNormalizeExpr']

    linBatch=parms['linBatch']
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()
    
    covariates=pd.read_csv(local+'data/covars.hipp.208.txt',sep='\t',header=None,index_col=None,names=[
        'intercept','sex','batch']).set_index(np.array(mouseIds))

    snps=pd.read_csv(local+'data/'+parms['snpFile'],sep='\t',header=None,index_col=None)
    snps.columns=['chr','Mbp','minor','major']+allIds
    snpData=snps.loc[snps['chr'].isin(snpChr),['chr','Mbp']]    
    snps=snps.loc[snps['chr'].isin(snpChr),['Mbp','minor','major']+mouseIds]
            
    np.savetxt(local+name+'process/dummy.txt',np.ones([len(mouseIds),1]),delimiter='\t')
    DBUpload(name+'process/dummy.txt',parms,toPickle=False)
    
    for snp in snpChr:
        genGRMHelp(snp,snps.loc[snpData['chr'].values.flatten()!=snp,:],parms)            
    if allChrGRM:
        genGRMHelp('all',snps,parms)

    print('grm finished',flush=True)
    snps=snps[['Mbp','minor','major']+mouseIds]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':'chr'+mouseGenes['chrom'].astype(str),
        'Mbp':(mouseGenes['cds_start']+mouseGenes['cds_end'])/2})    
    traitData=traitData[traitData['chr'].isin(traitChr)]    
    traitData=(pd.DataFrame({'trait':traits.columns}).merge(traitData,on='trait'))[['trait','chr','Mbp']]
    
    traits=traits[traitData['trait']].values
    
    if quantNormalizeExpr:
        traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))   

    if not linBatch:
        covariates=pd.concat([covariates,pd.get_dummies(covariates['batch'],drop_first=True)],axis=1).drop(columns='batch')
    
    DBWrite(mouseIds,name+'process/mouseIds',parms,toPickle=True)
    DBWrite(allIds,name+'process/allIds',parms,toPickle=True)
    
    if remCovFromTraits:
        reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(covariates,traits)
        traits=(traits-reg.predict(covariates))    
    
    if remPCFromTraits or remPCCorrSnp or PCIsPreds:
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

        DBWrite(R2.flatten(),name+'process/snpR2',parms,toPickle=True)
        
    if PCIsPreds:
        np.savetxt(local+name+'process/preds.txt',PCs,delimiter='\t')
    elif CovIsPreds:
        np.savetxt(local+name+'process/preds.txt',covariates,delimiter='\t')
    else:
        np.savetxt(local+name+'process/preds.txt',np.ones([len(mouseIds),1]),delimiter='\t')    
        
    DBUpload(name+'process/preds.txt',parms,toPickle=False)

    DBWrite(traitData,name+'process/traitData',parms,toPickle=True)
    DBWrite(np.array([traitData.shape[0]]),name+'process/N',parms,toPickle=True)

    traitCorr=np.corrcoef(traits,rowvar=False)
    LTraitCorr=makePSD(traitCorr)
    DBWrite(LTraitCorr,name+'process/LTraitCorr',parms,toPickle=True)
    
    DBWrite(traits,name+'process/traits',parms,toPickle=True)
    for trait in traitChr:
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',traits[:,traitData['chr'].values.flatten()==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
                    
    DBWrite(snpData,name+'process/snpData',parms,toPickle=True)
    
    for snp in parms['snpChr']:
        snps.loc[snpData['chr'].values.flatten()==snp,:].to_csv(local+name+'process/geno-'+snp+'.txt',sep='\t',
            index=False,header=False)        
        DBUpload(name+'process/geno-'+snp+'.txt',parms,toPickle=False)
        
    return()

def genGRMHelp(snp,snps,parms):
    local=parms['local']
    name=parms['name']
    grmParm=parms['grmParm']
    
    if DBIsFile(name+'process','grm-'+snp+'.txt',parms):
        return()
    
    snps.to_csv(local+name+'process/geno-grm-'+snp+'.txt',index=None,header=None,sep='\t')
    
    # generate loco
    print('grm',snp,flush=True)
    cmd=['./gemma','-g',local+name+'process/geno-grm-'+snp+'.txt','-gk',('1' if grmParm=='c' else '2'),
        '-o',name[:-1]+'-grm-'+snp,'-p',local+name+'process/dummy.txt']
    subprocess.run(cmd)

    # move grm to scratch
    os.remove(local+name+'process/geno-grm-'+snp+'.txt')
    
    os.rename('output/'+name[:-1]+'-grm-'+snp+'.'+grmParm+'XX.txt',local+name+'process/grm-'+snp+'.txt')
        
    DBUpload(name+'process/grm-'+snp+'.txt',parms,toPickle=False)

    return()