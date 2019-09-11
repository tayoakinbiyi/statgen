from ail.opPython.DB import *
import numpy as np
from scipy.stats import norm, beta, bernoulli
from ail.dataPrepPython.process import *
from ail.genPython.makePSD import *
from ail.plotPython.plotCorr import *

# 3)
def genH0Y(parms):
    etaGRM=parms['etaGRM']
    etaError=parms['etaError']
    traitChr=parms['traitChr']
    H1Chr=parms['H1Chr']
    name=parms['name']

    H1SnpSet=DBRead(name+'process/H1SnpSet.txt',parms,toPickle=False)
    genGRMHelp(H1Chr,H1SnpSet,parms,mouseLoc=None)
    
    LH1=makePSD(DBRead(name+'process/grm-'+H1Chr,parms,toPickle=False))
    LgrmAll=makePSD(DBRead(name+'process/grm-all',parms,toPickle=False))
    
    _,__=plotCorr({'All But Chr-'+H1Chr:LgrmAll,'Just H1Snps':LH1},'GRM Plot',parms)
    
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    H0Y=etaGRM**2*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitCorr.shape)),LTraitCorr.T)+etaError**2*np.matmul(
        norm.rvs(size=traitCorr.shape),LTraitCorr.T)
    
    DBWrite(H0Y,name+'process/H0Y',parms,toPickle=True)

    H0YCorr=np.corrcoef(H0Y,rowvar=False)
    LH0YCorr=makePSD(H0YCorr)
    DBWrite(LH0YCorr,name+'process/LH0YCorr',parms)

    for trait in traitChr:
        np.savetxt(local+name+'process/pheno-'+trait+'.txt',H0Y[:,traitData['chr']==trait],delimiter='\t')      
        DBUpload(name+'process/pheno-'+trait+'.txt',parms,toPickle=False)
    
    return()