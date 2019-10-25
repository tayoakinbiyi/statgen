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

def makePedFiles(parms):
    response=parms['response']
    allChrGRM=parms['allChrGRM']
    numCores=parms['numCores']
    local=parms['local']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']    
    PCIsPreds=parms['PCIsPreds']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    linBatch=parms['linBatch']
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()
    
    ################################################### make covar files ###################################################
    
    covariates=pd.read_csv(local+'data/covars.hipp.208.txt',sep='\t',header=None,index_col=None,names=['intercept','sex','batch'])
    
    if not linBatch:
        covariates=pd.concat([covariates[['sex']],pd.get_dummies(covariates['batch'],drop_first=True)],axis=1)
        
    if PCIsPreds:
        subprocess.call(['python2.7','ga/compute_auto_pcs'])
        reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(covariates,traits)
        traits=(traits-reg.predict(covariates))         
    else: # cov is preds
        pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(covariates))}),covariates.iloc[:,1:]],axis=1).to_csv(
            'ped/cov.phe',header=False,index=False)
    
    ################################################### make trait pheno file ###################################################
        
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

    if not DBIsFile('ped/ail.phe',parms):
        traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':'chr'+mouseGenes['chrom'].astype(str),
            'Mbp':(mouseGenes['cds_start']+mouseGenes['cds_end'])/2})    
        traitData=traitData.loc[:,traitData['chr'].isin(traitChr)]
        DBWrite(traitData,'ped/traitData',parms)

        traits=traits[traitData['trait']]
        if quantNormalizeExpr:
            traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))   

        pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits))}),traits],axis=1).to_csv(
            'ped/ail.phe',header=False,index=False)
            
    ################################################### make snp ped ###################################################

    snps=pd.read_csv(local+'data/ail.genos.ATGC.gwasSNPs.txt',sep='\t',header=None,index_col=None)
    snps.columns=['chr','Mbp','Minor','Major']+allIds
    chromosome=snps.iloc[:,0].str.slice(3)
    Mbp=snps.iloc[:,1]
    snps=snps[mouseIds].T

    if not DBIsFile('ped/ail.ped',parms):        
        bimSnps=snps        
        bimSnps.insert(0,'Phenotype',0)
        bimSnps.insert(0,'Sex',0)
        bimSnps.insert(0,'Maternal ID',0)
        bimSnps.insert(0,'Paternal ID',0)
        bimSnps.insert(0,'Individual ID',range(len(snps)))
        bimSnps.insert(0,'Family ID',0)
        
        snpData=pd.DataFrame({'chr':chromosome,'ID':range(len(snps)),'Mbp':Mbp})       
        
        bimSnps.to_csv('ped/ail.ped',header=False,index=False,sep='\t')   
        snpData.to_csv('ped/ail.map',header=False,index=False,sep='\t')    
        
        for snp in snpChr:         
            bimSnps[['Family ID','Individual ID','Paternal ID','Maternal ID','Sex','Phenotype']+
                bimSnps.columns[snpData['chr']==snp]].to_csv('ped/ail-'+snp+'.ped',header=False,index=False,sep='\t')
            snpData[snpData['chr']==snp].to_csv('ped/ail-'+snp+'.map',header=False,index=False,sep='\t')
    
    pdb.set_trace()
    ################################################### GRM ###################################################
    
    for snp in snpChr:         
        if not DBIsFile('ped/eigen-'+snp,parms):
            snpData.loc[snpData['chr']==snp,'ID'].to_csv('ped/extract-'+snp,header=False,index=False)
            snpData.loc[snpData['chr']!=snp,'ID'].to_csv('ped/extractSim-'+snp,header=False,index=False)
            
            cmd=[local+'ext/fastlmmc','-file','ped/ail-'+snp,'-fileSim','ped/ail','-covar','ped/cov.phe','-runGwasType','NORUN',
                 '-eigenOut','ped/eigen-'+snp,'-extractSim','ped/extractSim-'+snp,'-maxThreads',str(numCores),'-simOut','ped/grm-'+snp]
            subprocess.call(cmd)
            
    if not DBIsFile('ped/eigen-all',parms):
        DBCreateFolder('ped/eigen-all',parms)

        cmd=[local+'ext/fastlmmc','-file','ped/ail','-fileSim','ped/ail','-covar','ped/cov.phe','-runGwasType','NORUN',
             '-eigenOut','ped/eigen-all','-maxThreads',str(numCores),'-simOut','ped/grm-'+snp]
        subprocess.call(cmd)

        grm=pd.read_csv('ped/grm-all',sep='\t',index_col=0,header=0).values
        DBWrite(makePSD(grm),'LZCorr/Lgrm-all',parms)

    ################################################### Trait Corr ###################################################
    
    if not DBIsFile('LZCorr/LTraitCorr',parms):
        traitCorr=np.corrcoef(traits,rowvar=False)
        LTraitCorr=makePSD(traitCorr)
        DBWrite(LTraitCorr,'LZCorr/LTraitCorr',parms)
                
    return()
