import pandas as pd
import numpy as np
import pdb
import pyreadr

def getRealTraits(parms):
    response=parms['response']
    local=parms['local']
    numTraits=parms['parms'][2]
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

    robjects=pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[(mouseGenes['chrom'].isin([str(x) for x in range(1,22)]))&(mouseGenes['gene_name'].isin(traits.columns))]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
    traitData=traitData.sort_values(by=['chr','trait']).iloc[:numTraits]
    traitData.to_csv('ped/traitData',index=False,sep='\t')    
    traits=traits[traitData.trait].values
    
    return(traits)
