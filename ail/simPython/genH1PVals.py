from ail.opPython.DB import *
from ail.simPython.getPVals import *
from ail.simPython.makePower import *
import numpy as np

def genH1PVals(parms):
    name=parms['name']

    maxPower=parms['maxPower']
    minPower=parms['minPower']
    
    epsRange=parms['epsRange']
    muRange=parms['muRange']
            
    for eps in epsRange:
        for mu in muRange:
            if DBIsFile(name+'sim/pvals-'+str(mu)+'-'+str(eps),parms):
                continue

            DBWrite(pd.DataFrame(),name+'sim/pvals-'+str(mu)+'-'+str(eps),parms,toPickle=True)
            z=DBRead(name+'score/z-'+str(mu)+'-'+str(eps),parms,toPickle=True)

            pvalsMuEps=getPVals(z,mu,eps,parms)

            DBWrite(pvalsMuEps,name+'sim/pvals-'+str(eps)+'-'+str(mu),parms,toPickle=True)
            
    return()