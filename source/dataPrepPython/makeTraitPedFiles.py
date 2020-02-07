import numpy as np
import pandas as pd
from genPython.makePSD import *
import pdb

def makeTraitPedFiles(Y,traitData,parms):
    traitChr=parms['traitChr']
    
    for trait in traitChr:
        np.savetxt('ped/Y-'+str(trait)+'.txt',Y[:,traitData['chr']==trait],delimiter='\t')
        pd.concat([pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0}),pd.DataFrame(Y[:,traitData['chr']==trait])],
            axis=1).to_csv('ped/Y-'+str(trait)+'.phe',header=False,index=False,sep='\t')
    
    np.savetxt('ped/Y.txt',Y,delimiter='\t')
    pd.concat([pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0}),pd.DataFrame(Y)],axis=1).to_csv('ped/Y.phe',
        header=False,index=False,sep='\t')

    YCorr=np.corrcoef(Y,rowvar=False)
    np.savetxt('LZCorr/LTraitCorr',makePSD(YCorr),delimiter='\t')

    return()