from ail.opPython.DB import *
import numpy as np
from scipy.stats import norm, beta, bernoulli
from ail.dataPrepPython.process import *
from ail.genPython.makePSD import *
from ail.plotPython.plotCorr import *

def genNullY(parms):
    etaGRM=parms['etaGRM']
    etaError=parms['etaError']
    traitChr=parms['traitChr']
    name=parms['name']
    local=parms['local']

    DBLocalWrite(DBRead(name+'process/grm-all.txt',parms,toPickle=False),name+'process/grm-all.txt',parms,toPickle=False)
    grm=np.loadtxt(local+name+'process/grm-all.txt',delimiter='\t')
    LgrmAll=makePSD(grm) # remove all thse PSD
    DBWrite(LgrmAll,name+'process/LgrmAll',parms,toPickle=True)
    
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)
    LTraitCorr=DBRead(name+'process/LTraitCorr',parms,toPickle=True)
    traitSize=[len(LgrmAll),len(LTraitCorr)]

    NullY=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LTraitCorr.T)
    
    DBWrite(NullY,name+'process/NullY',parms,toPickle=True)

    NullYCorr=np.corrcoef(NullY,rowvar=False)
    LNullYCorr=makePSD(NullYCorr)
    DBWrite(LNullYCorr,name+'process/LNullYCorr',parms,toPickle=True)

    LNullYGRMCorr=makePSD(np.corrcoef(NullY,rowvar=True))
    DBWrite(LNullYGRMCorr,name+'process/LNullYGRMCorr',parms,toPickle=True)
    
    plotCorr({'NullY':'process/LNullYCorr','YRaw':'process/LTraitCorr'},parms)
    plotCorr({'NullYGRM':'process/LNullYGRMCorr','GRMAll':'process/LgrmAll'},parms)

    for trait in traitChr:
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',NullY[:,traitData['chr']==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
    
    return()