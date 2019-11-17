import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from ail.statsPython.ELL import *
from ail.statsPython.cpma import *
from ail.opPython.DB import *
from ail.statsPython.noCorrStats import *
from ail.statsPython.gbj import *

def MCMCStats(z,pval,nameParm,parms,ELLAll,offDiag,Types):
    ellDSet=parms['ellDSet']
    name=parms['name']
    local=parms['local']
    N=z.shape[1]
        
    stats=[]
    
    if ('cpma' in Types):
        stats+=[cpma(pval,nameParm,parms)]
        
    if ('noCorr' in Types):
        stats+=[noCorrStats(z,pval,nameParm,parms)]

    for dParm in ellDSet:
        Type='ell-'+str(dParm)
        if (Type in Types):        
            stats+=[ELL(pval,nameParm,dParm,parms,ELLAll)]
    
    if ('gbj' in Types):
        stats+[gbj(z,pval,nameParm,parms,offDiag)]
    
    return(pd.concat(stats,axis=0))
