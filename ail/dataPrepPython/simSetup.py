import pandas as pd
import numpy as np
import pdb
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pyreadr
import subprocess
import shutil

from ail.opPython.DB import *
from ail.genPython.makePSD import *
from ail.dataPrepPython.genGRM import *

def simSetup(parms):
    print('simSetup')
    
    response=parms['response']
    name=parms['name']
    local=parms['local']
    linBatch=parms['linBatch']
    traitChr=parms['traitChr']
    grmSnpChr=parms['grmSnpChr']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    H0SnpSize=parms['H0SnpSize']
    H1SnpSize=parms['H1SnpSize']
        
    DBSyncLocal('data',parms)
        
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()
    
    covariates=pd.read_csv(local+'data/covars.hipp.208.txt',sep='\t',header=None,index_col=None,names=[
        'Intercept','sex','batch']).set_index(np.array(mouseIds))

    snps=pd.read_csv(local+'data/'+parms['snpFile'],sep='\t',header=None,index_col=None)
    snps.columns=['chr','Mbp','minor','major']+allIds
    snps=snps.loc[snps['chr'].isin(grmSnpChr),['Mbp','minor','major']+mouseIds]
    
    np.savetxt(local+name+'process/dummy.txt',np.ones([len(mouseIds),1]),delimiter='\t')
    DBUpload(name+'process/dummy.txt',parms,toPickle=False)
    
    genGRM('all',snps,parms)
    shutil.copy2(local+name+'process/grm-all.txt',local+name+'process/grm-chr0.txt')
    shutil.copy2(local+name+'process/grm-all.txt',local+name+'process/grm-chr1.txt')
    DBUpload(name+'process/grm-chr0.txt',parms,toPickle=False)
    DBUpload(name+'process/grm-chr1.txt',parms,toPickle=False)

    print('grm finished',flush=True)

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':'chr'+mouseGenes['chrom'].astype(str),
        'Mbp':(mouseGenes['cds_start']+mouseGenes['cds_end'])/2})    
    traitData=traitData[traitData['chr'].isin(traitChr)]    
    traitData=(pd.DataFrame({'trait':traits.columns}).merge(traitData,on='trait'))[['trait','chr','Mbp']]
    
    traits=traits[traitData['trait']].values
    
    if quantNormalizeExpr:
        traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))   

    if not linBatch:
        covariates=pd.concat([covariates[['Intercept','sex']],pd.get_dummies(covariates['batch'],drop_first=True)],axis=1)
    
    DBWrite(mouseIds,name+'process/mouseIds',parms,toPickle=True)
    DBWrite(allIds,name+'process/allIds',parms,toPickle=True)
    
    reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(covariates,traits)
    traits=(traits-reg.predict(covariates))    
    
    np.savetxt(local+name+'process/preds.txt',np.ones([len(mouseIds),1]),delimiter='\t')            
    DBUpload(name+'process/preds.txt',parms,toPickle=False)

    DBWrite(traitData,name+'process/traitData',parms,toPickle=True)
    DBWrite(np.array([traitData.shape[0]]),name+'process/N',parms,toPickle=True)

    traitCorr=np.corrcoef(traits,rowvar=False)
    LTraitCorr=makePSD(traitCorr)
    DBWrite(LTraitCorr,name+'process/LTraitCorr',parms,toPickle=True)
    
    for trait in traitChr:
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',traits[:,traitData['chr']==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
        
    return()
