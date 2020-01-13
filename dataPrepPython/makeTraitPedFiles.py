import numpy as np
import pandas as pd
from genPython.makePSD import *
import pdb

def makeTraitPedFiles(Y,traitData,mouseIds,parms):
    traitChr=parms['traitChr']
    N=Y.shape[0]
    
    for trait in traitChr:
        np.savetxt('ped/Y-'+str(trait)+'.txt',Y[:,traitData['chr']==trait],delimiter='\t')
        pd.concat([pd.DataFrame({'Family ID':mouseIds,'Individual ID':0}),pd.DataFrame(Y[:,traitData['chr']==trait])],axis=1).to_csv(
            'ped/Y-'+str(trait)+'.phe',header=False,index=False,sep='\t')
    
    YCorr=np.corrcoef(Y,rowvar=False)
    np.savetxt('LZCorr/LTraitCorr',makePSD(YCorr),delimiter='\t')

    return()