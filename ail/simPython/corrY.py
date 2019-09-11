import pdb
from ail.dataPrepPython.process import *

# 1)
def corrY(parms):
    H1Chr=parms['H1Chr']
    parms={
        **parms,
        'response':'hipRaw',
        'quantNormalizeExpr':True,
        'remPCFromSnp':False,
        'remPCFromTraits':False,
        'remPCCorrSnp':False,
        'PCIsPreds':False,
        'CovIsPreds':False,
        'remCovFromTraits':True,
        'grmParm':'s',
        'wald':True,
        'subsetFirstGRM':True,
        'allChrGRM':True
    }
    
    process(parms)
    
    return()