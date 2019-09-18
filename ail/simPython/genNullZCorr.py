from ail.opPython.DB import *
import numpy as np
from ail.plotPython.plotCorr import *
from ail.dataPrepPython.genMeans import *
from ail.dataPrepPython.genCorrMats import *
from ail.dataPrepPython.genCorr import *

def genNullZCorr(parms):
    name=parms['name']
    
    genMeans(parms)
    genCorrMats(parms)
    genCorr('all',parms)
    
    LTraitCorr=DBRead(name+'process/LTraitCorr',parms,toPickle=True)
    LNullYCorr=DBRead(name+'process/LNullYCorr',parms,toPickle=True)
    LZCorr=DBRead(name+'corr/LZCorr-all',parms,toPickle=True)
    
    _,__=plotCorr({'Raw Y':'process/LTraitCorr','Null Y':'process/LNullYCorr'},parms) # make plotCorr multiply L,L.T
    _,offDiag=plotCorr({'Raw Y':'process/LTraitCorr','Null Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T
    _,__=plotCorr({'Null Y':'process/LNullYCorr','Null Z':'corr/LZCorr-all'},parms) # make plotCorr multiply L,L.T

    DBWrite(offDiag,name+'sim/NullZPairwiseCors',parms,toPickle=True)
    
    return()