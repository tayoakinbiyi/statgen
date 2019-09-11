from ail.statsPython.pvalsMCMCStats import *
from ail.statsPython.ghc import *
from ail.statsPython.gbj import *

import pandas as pd

def getPVals(Z,parms)
    pvalCutOff=parms['pvalCutOff']

    pvals=pd.DataFrame()
    pvals=pd.concat([pvals,pvalsMCMCStats(Z,parms)],axis=1)# find pvals from MCMCStats and MCMCIIDStats
    pvals=pd.concat([pvals,ghc(z,parms)],axis=1)
    pvals=pd.concat([pvals,gbj(z,{**parms,'gbjName':str(parms['mu'])+'-'+str(parms['epsilon'])})],axis=1)

    return((pvals<pvalCutOff).sum(axis=0))
    # save pvals to DB
    #stat functions return Series named after stat or DF with each column named for the stat