import numpy as np
import pandas as pd
import pyreadr
import pdb

def initSnpTraits(parms,snps):
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    response=parms['response']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    local=parms['local']
    chrLoc=parms['chrLoc'] if parms['chrLoc'] is not None else range(2000)
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int)
    np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[mouseGenes['chrom'].isin([str(x) for x in traitChr])]

    N=traits.shape[1]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
    traitData=traitData.loc[traitData['trait'].isin(traits.columns)]
    traitData=traitData.groupby('chr').apply(lambda df: pd.concat([df.reset_index(drop=True),
        pd.DataFrame({'chrLoc':range(len(df))})],axis=1)).reset_index(drop=True)
    traitData=traitData.sort_values(by=['chr','chrLoc'])
    traitData=traitData[traitData['chrLoc'].isin(chrLoc)]
    traitData.insert(0,'loc',range(len(traitData)))
    
    traitData.to_csv('ped/traitData',index=False,sep='\t')
    
    traits=traits[traitData.trait].values
        
    if quantNormalizeExpr:
        traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))
    
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
    
    return(traits,traitData,snps,snpData,mouseIds)