from ail.opPython.DB import *
import numpy as np
from scipy.stats import norm, beta, bernoulli
from ail.dataPrepPython.process import *
from ail.genPython.makePSD import *

from ail.plotPython.plotCorr import *

def genY(parms):
    print('genY',flush=True)
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

    Y=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LTraitCorr.T)
    
    DBWrite(Y,name+'process/Y',parms,toPickle=True)

    YCorr=np.corrcoef(Y,rowvar=False)
    LYCorr=makePSD(YCorr)
    DBWrite(LYCorr,name+'process/LYCorr',parms,toPickle=True)

    LYGRMCorr=makePSD(np.corrcoef(Y,rowvar=True))
    DBWrite(LYGRMCorr,name+'process/LYGRMCorr',parms,toPickle=True)

    for trait in traitChr:
        print('genY trait '+trait,flush=True)
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',Y[:,traitData['chr']==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
    
    DBCreateFolder(name,'plotCorr',parms)
    plotCorr({'Y':'process/LYCorr','YRaw':'process/LTraitCorr'},parms)
    plotCorr({'YGRM':'process/LYGRMCorr','GRMAll':'process/LgrmAll'},parms)

    return()