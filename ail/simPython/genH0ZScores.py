from ail.opPython.DB import *
import numpy as np
from ail.plotPython.plotCorr import *


def genH0ZScores(parms):
    name=parms['name']
    
    genMeans(parms)
    genCorrMats(parms)
    genCorr('all',parms)
    
    LTraitCorr=DBRead(name+'process/LTraitCorr',parms,toPickle=True)
    LH0YCorr=DBRead(name+'process/LH0YCorr',parms,toPickle=True)
    LZCorr=DBRead(name+'process/LZCorr-all',parms,toPickle=True)
    
    _,__=plotCorr({'Raw Y':LTraitCorr,'H0 Y':LH0YCorr},'Raw vs H0 Y',parms) # make plotCorr multiply L,L.T
    _,offDiag=plotCorr({'Raw Y':LTraitCorr,'H0 Z':LZCorr},'Raw vs H0 Z',parms) # make plotCorr multiply L,L.T
    _,__=plotCorr({'H0 Y':LH0YCorr,'H0 Z':LZCorr},'H0 Y vs H0 Z',parms) # make plotCorr multiply L,L.T

    DBWrite(offDiag,name+'sim/H0ZPairwiseCors',parms,toPickle=True)
    
    return()