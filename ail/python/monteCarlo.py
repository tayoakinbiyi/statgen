import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from python.ggof import *
from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *
from python.fitStat import *

def monteCarlo(L,eps,mu,repKey,ebb,var,parms):
    N=parms['N']
    Reps=parms[repKey]
    cpus=parms['cpus']
    
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,Reps))).T
    
    if mu*eps>0:
        z[:,range(eps)]+=mu
    
    print(eps,mu,psutil.virtual_memory().percent)

    power,fail=fitStat(z,ebb,var,parms)
    
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)
    
    if len(fail)>0:
        fail.index=len(fail)*[0]
        fail=fail.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)

    return(power,fail)

