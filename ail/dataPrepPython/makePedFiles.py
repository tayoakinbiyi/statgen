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

from multiprocessing import cpu_count

import json

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
    
    DBCreateFolder('ped',parms)
    DBCreateFolder('LZCorr',parms)
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()
    np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()
    
    ################################################### make covar files ###################################################
    
    if not os.path.exists('ped/cov.phe'):
        covariates=pd.read_csv(local+'data/covars.hipp.208.txt',sep='\t',header=None,index_col=None,
            names=['intercept','sex','batch'])

        if not linBatch:
            covariates=pd.concat([covariates[['sex']],pd.get_dummies(covariates['batch'],drop_first=True)],axis=1)

        if PCIsPreds:
            subprocess.call(['python2.7','ga/compute_auto_pcs'])
            reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(covariates,traits)
            traits=(traits-reg.predict(covariates))         
        else: # cov is preds
            pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(covariates))}),covariates],axis=1).to_csv(
                'ped/cov.phe',header=False,index=False,sep='\t')
    
    ################################################### make trait pheno file ###################################################
        
    if not os.path.exists('ped/ail.phe'):
        traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0).reset_index()

        traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(str),
            'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})   
        traitData=traitData.loc[traitData['chr'].isin(traitChr),:]
        traitData=traitData.loc[traitData['trait'].isin(traits.columns)].reset_index()
        traitData.to_csv('ped/traitData',index=False,sep='\t')

        traits=traits.loc[:,traitData['trait']]
        if quantNormalizeExpr:
            traits=pd.DataFrame(norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1)))

        pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits))}),traits],axis=1).to_csv(
            'ped/ail.phe',header=False,index=False,sep='\t')

        traitCorr=np.corrcoef(traits,rowvar=False)
        LTraitCorr=makePSD(traitCorr)
        np.savetxt('LZCorr/LTraitCorr',LTraitCorr,delimiter='\t')
        np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')
        
        DBLog('number of traits by chromosome \n'+json.dumps(traitData.groupby('chr')['trait'].count().to_dict(),indent=3),parms)

    ################################################### make snp ped ###################################################

    if not os.path.exists('ped/ail.ped'):
        snps=pd.read_csv(local+'data/ail.genos.ATGC.gwasSNPs.txt',sep='\t',header=None,index_col=None)
        snps.columns=['chr','Mbp','Minor','Major']+allIds
        chromosome=snps.iloc[:,0].str.slice(3).astype(str)
        Mbp=snps.iloc[:,1]
        snps=snps[mouseIds].reset_index(drop=True).T

        bimSnps=snps        
        bimSnps.insert(0,'Phenotype',0)
        bimSnps.insert(0,'Sex',0)
        bimSnps.insert(0,'Maternal ID',0)
        bimSnps.insert(0,'Paternal ID',0)
        bimSnps.insert(0,'Individual ID',range(len(snps)))
        bimSnps.insert(0,'Family ID',0)

        snpData=pd.DataFrame({'chr':chromosome,'ID':range(len(chromosome)),'genetic dist':0,'Mbp':Mbp})       

        bimSnps.to_csv('ped/ail.ped',header=False,index=False,sep='\t')   
        snpData.to_csv('ped/ail.map',header=False,index=False,sep='\t')    
        
        snpData[['chr','ID','Mbp']].to_csv('ped/snpData',index=False,sep='\t')

        for snp in snpChr:   
            if os.exists('ped/ail-'+snp+'.ped'):
                continue

            bimSnps[['Family ID','Individual ID','Paternal ID','Maternal ID','Sex','Phenotype']+
                snpData[snpData['chr']==snp].index.tolist()].to_csv('ped/ail-'+snp+'.ped',header=False,index=False,sep='\t')
            snpData[snpData['chr']==snp].to_csv('ped/ail-'+snp+'.map',header=False,index=False,sep='\t')

        DBLog('number of snps by chromosome \n'+json.dumps(snpData.groupby('chr')['ID'].count().to_dict(),indent=3),parms)            
    else:
        snpData=pd.read_csv('ped/snpData',header=0,index_col=None,sep='\t')
    
    
    ################################################### GRM ###################################################
    
    for snp in snpChr:  
        if os.path.exists('ped/eigen-'+snp):
            continue
            
        snpData.loc[snpData['chr']!=snp,'ID'].to_csv('ped/extractSim-'+snp,header=False,index=False)

        cmd=[local+'ext/fastlmmc','-file','ped/ail','-fileSim','ped/ail','-pheno','ped/ail.phe','-covar','ped/cov.phe',
             '-runGwasType','NORUN','-eigenOut','ped/eigen-'+snp,'-extractSim','ped/extractSim-'+snp,'-maxThreads',
             str(cpu_count()),'-simOut','ped/grm-'+snp]
        subprocess.call(cmd)
    
    if not os.path.exists('ped/eigen-all'):
        cmd=[local+'ext/fastlmmc','-file','ped/ail','-fileSim','ped/ail','-pheno','ped/ail.phe','-covar','ped/cov.phe',
             '-runGwasType','NORUN','-eigenOut','ped/eigen-all','-maxThreads',str(cpu_count()),'-simOut','ped/grm-all']
        subprocess.call(cmd)

        grm=pd.read_csv('ped/grm-all',sep='\t',index_col=0,header=0).values
        np.savetxt('LZCorr/Lgrm-all',makePSD(grm),delimiter='\t')

                
    return()
