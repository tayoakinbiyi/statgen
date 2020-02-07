import numpy as np
import pandas as pd
import pyreadr
import pdb
from genPython.makePSD import *

def initSnpTraits(parms,snps):
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    response=parms['response']
    local=parms['local']

    traitSubset=parms['traitSubset'] if parms['traitSubset'] is not None else range(2000)
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[mouseGenes['chrom'].isin([str(x) for x in traitChr])]

    N=traits.shape[1]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
    traitData=traitData.loc[traitData['trait'].isin(traits.columns)]
    traitData=traitData.groupby('chr').apply(lambda df: pd.concat([df.reset_index(drop=True),
        pd.DataFrame({'traitSubset':range(len(df))})],axis=1)).reset_index(drop=True)
    traitData=traitData.sort_values(by=['chr','traitSubset'])
    traitData=traitData[traitData['traitSubset'].isin(traitSubset)]
    traitData.insert(0,'loc',range(len(traitData)))
    traitDataMat=[]
    for trait in traitChr:
        traitDataMat+=[traitData[traitData['chr']==trait]]
    traitData=pd.concat(traitDataMat,axis=0)
    
    traitData.to_csv('ped/traitData',index=False,sep='\t')
    
    traits=traits[traitData.trait].values
        
    LTraitCorr=makePSD(np.corrcoef(traits,rowvar=False))        
    np.savetxt('LZCorr/LTraitCorr',LTraitCorr,delimiter='\t')
    
    if snps:
        snps=pd.read_csv(local+'data/snps.txt',sep='\t',header=0,index_col=None)
        snps=snps[snps['chr'].str.slice(3).astype(int).isin(snpChr)].reset_index(drop=True)
        chromosome=snps['chr'].str.slice(3).astype(int).values.flatten()
        Mbp=snps['Mbp'].values.flatten()
        snps=snps.iloc[:,2:].T.reset_index(drop=True)

        snpData=pd.DataFrame({'chr':chromosome,'ID':range(len(chromosome)),'genetic dist':0,'Mbp':Mbp})   
    else:
        snps=None
        snpData=None
    
    return(traits,traitData,snps,snpData)