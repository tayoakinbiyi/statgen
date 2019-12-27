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

def fastVsGemmaPed(parms):
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
    mouseGenes=mouseGenes[mouseGenes['chrom'].isin([str(x) for x in traitChr])]

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()
    
    ################################################### make covar files ###################################################
    
    pd.DataFrame({'Family ID':0,'Individual ID':range(len(mouseIds)),'intercept':1}).to_csv(
        'ped/cov.phe',header=False,index=False,sep='\t')
    
    ################################################### make trait pheno file ###################################################
        
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0).reset_index()

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'],
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})   
    traitData=traitData.loc[traitData['trait'].isin(traits.columns)]
    traitData.to_csv('ped/traitData',index=False,sep='\t')

    traits=traits.loc[:,traitData['trait']]

    pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits))}),traits],axis=1).to_csv(
        'ped/ail.phe',header=False,index=False,sep='\t')

    np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')

    DBLog('number of traits by chromosome \n'+json.dumps(traitData.groupby('chr')['trait'].count().to_dict(),indent=3),parms)

    ################################################### make snp ped ###################################################

    snps=pd.read_csv(local+'data/ail.genos.ATGC.gwasSNPs.txt',sep='\t',header=None,index_col=None)
    snps.columns=['chr','Mbp','Minor','Major']+allIds
    chromosome=snps.iloc[:,0].str.slice(3).astype(int)
    Mbp=snps.iloc[:,1]
    snps=snps.loc[chromosome.isin(snpChr),mouseIds].T.reset_index(drop=True)
    snpData=pd.DataFrame({'chr':1,'ID':range(snps.shape[1]),'genetic dist':0,'Mbp':1})       
    
    snps=pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(snps)),'Paternal ID':0,'Maternal ID':0,'Sex':0,
        'Phenotype':0}),snps],axis=1)

    snps.to_csv('ped/ail-1.ped',header=False,index=False,sep='\t')   
    snpData.to_csv('ped/ail-1.map',header=False,index=False,sep='\t')            
    snpData.to_csv('ped/snpData',index=False,sep='\t')
                
    return()
