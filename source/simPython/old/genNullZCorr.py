from ail.opPython.DB import *
import numpy as np
from ail.plotPython.plotCorr import *
from ail.dataPrepPython.genMeans import *
from ail.dataPrepPython.genCorrMats import *
from ail.dataPrepPython.genLZCorr import *

def genNullZCorr(parms):
    name=parms['name']
    local=parms['local']
    
    H0NameParm='mu:0-eps:0'

    genMeans(parms,H0NameParm)
    genCorrMats(parms,H0NameParm)
    genLZCorr('all',parms,H0NameParm)
    
    LTraitCorr=DBRead(name+'process/LTraitCorr',parms,toPickle=True)
    LYCorr=DBRead(name+'process/LYCorr',parms,toPickle=True)
    LZCorr=DBRead(name+'LZCorr/LZCorr-all',parms,toPickle=True)
    
    plotCorr({'Raw Y':'process/LTraitCorr','H0 Y':'process/LYCorr'},parms) # make plotCorr multiply L,L.T
    plotCorr({'Raw Y':'process/LTraitCorr','H0 Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T
    plotCorr({'H0 Y':'process/LYCorr','H0 Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T
    
    return()