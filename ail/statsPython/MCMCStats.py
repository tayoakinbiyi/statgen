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

def MCMCStats(z,pval,nameParm,parms,folder,ELLAll,offDiag):
    ellDSet=parms['ellDSet']
    name=parms['name']
    local=parms['local']
    Types=parms['Types']
    N=z.shape[1]
        
    if ('cpma' in Types):
        cpma(pval,nameParm,folder,parms)
        
    if ('bj' in Types):
        noCorrStats(z,pval,nameParm,parms,folder)

    for dParm in ellDSet:
        stat='ell-'+str(dParm)
        if (stat in Types):        
            ELL(pval,nameParm,dParm,parms,folder,ELLAll)    
    
    if ('gbj' in Types):
        gbj(z,pval,nameParm,parms,folder,offDiag)
      
    return()
