from ail.opPython.DB import *
import numpy as np
from ail.plotPython.plotCorr import *
from ail.dataPrepPython.genMeans import *
from ail.dataPrepPython.genCorrMats import *
from ail.dataPrepPython.genCorr import *

def genH0ZCorr(parms):
    name=parms['name']
    parms={**parms.copy(),'snpChr':['chr0','chr1']}
    
    genMeans(parms)
    genCorrMats(parms)
    genCorr('all',parms)
    
    LTraitCorr=DBRead(name+'process/LTraitCorr',parms,toPickle=True)
    LYCorr=DBRead(name+'process/LYCorr',parms,toPickle=True)
    LZCorr=DBRead(name+'corr/LZCorr-all',parms,toPickle=True)
    
    _,__=plotCorr({'Raw Y':'process/LTraitCorr','H0 Y':'process/LYCorr'},parms) # make plotCorr multiply L,L.T
    _,offDiag=plotCorr({'Raw Y':'process/LTraitCorr','H0 Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T
    _,__=plotCorr({'H0 Y':'process/LYCorr','H0 Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T

    DBWrite(offDiag,name+'sim/H0ZPairwiseCors',parms,toPickle=True)
    
    return()